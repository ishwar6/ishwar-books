"""Chapter 4, Section 4.4: parallelisation. Three small measurements with the real API (gpt-5-mini, reasoning_effort=minimal).
A. VOTING: ten multi-step word problems with one exact answer; five independent samples each; accuracy of the plurality vote (most frequent answer) over the
   first n samples for n = 1, 3, 5, and how often the samples agree.
B. BEST-OF-N WITH A VERIFIER: ten Game-of-24 puzzles (combine four numbers with + - * / into 24); a programmatic verifier checks each
   candidate expression; a puzzle counts as solved by n samples if any of the first n verifies.
C. LATENCY: five samples for one puzzle run one after another and then concurrently on five threads, timed with the wall clock, repeated
   over three puzzles. Cost is identical both ways; only the wall-clock time changes."""
import re, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from common import Log, save
import ch4_llm as llm

log = Log('ch4_parallel')
llm.set_log(log)
N = 5

VOTE = [
    ('A shop sells pens at 3 for 2.40 and notebooks at 1.75 each. Dana buys 9 pens and 4 notebooks and pays with a 20 note. How much change does she get?', '5.80'),
    ('A train leaves at 09:40, travels 210 km at 84 km/h, waits 25 minutes, then travels 90 km at 60 km/h. At what time (HH:MM) does it arrive?', '14:05'),
    ('A rectangle is 3 longer than twice its width and its perimeter is 48. What is its area?', '119'),
    ('A tank holds 350 litres and starts with 70. A pipe adds 12 litres a minute while a drain removes 5. After how many minutes is it full?', '40'),
    ('Prices rose 20% and then fell 25%. A jacket now costs 72. What was the original price?', '80'),
    ('Three friends split a bill of 138: Ana pays twice what Ben pays and Cy pays 18 more than Ben. How much does Ana pay?', '60'),
    ('A cyclist rides 36 km at 12 km/h and returns at 18 km/h. What is the average speed for the whole trip in km/h?', '14.4'),
    ('The sum of three consecutive even numbers is 150. What is the product of the smallest and the largest?', '2496'),
    ('A recipe for 6 people needs 450 g of flour. How many grams are needed for 16 people?', '1200'),
    ('What is the sum of all two-digit numbers divisible by 7?', '728'),
]
PUZZLES = [[1, 2, 3, 4], [6, 6, 6, 6], [3, 5, 7, 9], [2, 2, 2, 3], [2, 3, 4, 6], [2, 4, 6, 8], [4, 7, 8, 8], [1, 3, 4, 6], [3, 3, 8, 8], [1, 5, 5, 5]]
VOTE_PROMPT = 'Solve the problem. Reply with ONLY the final answer (a number, or a time as HH:MM), no explanation.\n\nProblem: {q}'
P24_PROMPT = ('Game of 24: using each of the numbers {nums} exactly once, with + - * / and parentheses, write one arithmetic expression that equals 24. '
              'Reply with ONLY the expression, nothing else.')


from ch4_grade import grade, numbers_in, _clean


def norm(s):
    """Canonical key for counting votes (one number, an HH:MM time, or the cleaned text). Grading uses ch4_grade.grade."""
    t = re.findall(r'\b\d{1,2}:\d{2}\b', s or '')
    if len(t) == 1:
        return t[0]
    ns = numbers_in(s)
    return f'{ns[0]:g}' if len(ns) == 1 else _clean(s)


def verify24(expr, nums):
    """True if expr uses exactly the given numbers once each, only + - * / ( ), and evaluates to 24."""
    e = (expr or '').strip().strip('`').replace('×', '*').replace('÷', '/').replace('x', '*')
    e = re.sub(r'\s*=\s*24\s*$', '', e)
    if not e or not re.fullmatch(r'[0-9+\-*/(). ]+', e):
        return False
    if sorted(int(n) for n in re.findall(r'\d+', e)) != sorted(nums):
        return False
    try:
        return abs(eval(e) - 24) < 1e-6                         # safe: the regex admits only digits, operators and brackets
    except Exception:
        return False


# ---- A. voting
log(f'== A. Voting: {len(VOTE)} word problems, {N} samples each ({llm.MODEL}, reasoning_effort=minimal) ==')
vote_rows = []
for q, gold in VOTE:
    samples = []
    for _ in range(N):
        t, u = llm.ask(VOTE_PROMPT.format(q=q), max_tokens=300)
        samples.append((norm(t), u))
    answers = [a for a, _ in samples]
    row = {'q': q, 'gold': gold, 'answers': answers, 'use': [u for _, u in samples]}
    for n in [1, 3, 5]:
        top, cnt = Counter(answers[:n]).most_common(1)[0]     # plurality: the most frequent answer; ties go to the earliest sample
        row[f'vote{n}'] = grade(top, gold)
        row[f'agree{n}'] = cnt / n
    vote_rows.append(row)
    log(f'  gold {gold:>6}  samples {answers}  vote@1 {"y" if row["vote1"] else "N"}  vote@3 {"y" if row["vote3"] else "N"}  vote@5 {"y" if row["vote5"] else "N"}')
vote_summary = {}
for n in [1, 3, 5]:
    acc = sum(r[f'vote{n}'] for r in vote_rows) / len(vote_rows)
    tot = {k: sum(u[k] for r in vote_rows for u in r['use'][:n]) for k in ['input', 'output', 'reasoning']}
    vote_summary[n] = {'acc': acc, 'cost_usd': llm.cost(tot), 'tokens': tot, 'mean_agreement': sum(r[f'agree{n}'] for r in vote_rows) / len(vote_rows)}
log('')
log(f'{"n":>3} {"accuracy":>9} {"cost $":>8} {"x n=1":>6} {"mean agreement":>15}')
for n, s in vote_summary.items():
    log(f'{n:>3} {s["acc"]:>9.0%} {s["cost_usd"]:>8.4f} {s["cost_usd"] / vote_summary[1]["cost_usd"]:>5.1f}x {s["mean_agreement"]:>15.0%}')
unanimous_wrong = sum(1 for r in vote_rows if len(set(r['answers'])) == 1 and not grade(r['answers'][0], r['gold']))
log(f'reading: a plurality vote helps only where the right answer is the most frequent one among the samples; {unanimous_wrong} problem(s) had five identical wrong answers, which no vote can fix.')

# ---- B. best-of-n with a programmatic verifier
log('')
log(f'== B. Best-of-n with a verifier: {len(PUZZLES)} Game-of-24 puzzles, {N} samples each ==')
p24_rows = []
for nums in PUZZLES:
    samples = []
    for _ in range(N):
        t, u = llm.ask(P24_PROMPT.format(nums=nums), max_tokens=200)
        samples.append((t.strip(), verify24(t, nums), u))
    row = {'nums': nums, 'exprs': [s for s, _, _ in samples], 'ok': [v for _, v, _ in samples], 'use': [u for _, _, u in samples]}
    for n in [1, 3, 5]:
        row[f'best{n}'] = any(row['ok'][:n])
    p24_rows.append(row)
    log(f'  {str(nums):<14} verified {"".join("y" if v else "." for v in row["ok"])}  best@1 {"y" if row["best1"] else "N"}  best@3 {"y" if row["best3"] else "N"}  '
        f'best@5 {"y" if row["best5"] else "N"}  e.g. {row["exprs"][0][:40]!r}')
p24_summary = {}
for n in [1, 3, 5]:
    acc = sum(r[f'best{n}'] for r in p24_rows) / len(p24_rows)
    tot = {k: sum(u[k] for r in p24_rows for u in r['use'][:n]) for k in ['input', 'output', 'reasoning']}
    p24_summary[n] = {'acc': acc, 'cost_usd': llm.cost(tot), 'tokens': tot}
per_sample = sum(sum(r['ok']) for r in p24_rows) / (len(p24_rows) * N)
log('')
log(f'{"n":>3} {"solved":>7} {"cost $":>8} {"x n=1":>6}  (per-sample pass rate p = {per_sample:.0%}; 1-(1-p)^n predicts {1 - (1 - per_sample) ** 3:.0%} at n=3 and {1 - (1 - per_sample) ** 5:.0%} at n=5)')
for n, s in p24_summary.items():
    log(f'{n:>3} {s["acc"]:>7.0%} {s["cost_usd"]:>8.4f} {s["cost_usd"] / p24_summary[1]["cost_usd"]:>5.1f}x')
log('reading: with a free verifier, n samples cost n times the tokens and the latency of one; the gain follows 1-(1-p)^n only if the samples are independent.')

# ---- C. latency: sequential against concurrent fan-out
log('')
log(f'== C. Wall-clock latency of {N} samples: one after another against {N} threads (three puzzles) ==')
lat = []
for nums in PUZZLES[6:9]:
    t0 = time.time()
    for _ in range(N):
        llm.ask(P24_PROMPT.format(nums=nums), max_tokens=200)
    seq = time.time() - t0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N) as ex:
        list(ex.map(lambda _: llm.ask(P24_PROMPT.format(nums=nums), max_tokens=200), range(N)))
    par = time.time() - t0
    lat.append({'nums': nums, 'sequential_s': seq, 'concurrent_s': par})
    log(f'  {str(nums):<14} sequential {seq:>5.1f} s   concurrent {par:>5.1f} s   speedup {seq / par:>4.1f}x')
ms, mp = sum(l['sequential_s'] for l in lat) / len(lat), sum(l['concurrent_s'] for l in lat) / len(lat)
log(f'mean: sequential {ms:.1f} s, concurrent {mp:.1f} s, speedup {ms / mp:.1f}x; the token bill is the same in both cases.')

U = llm.snapshot()
log('')
log(f'total spend: ${llm.cost(U):.4f} ({U["calls"]} calls, {U["input"]:,} input + {U["output"]:,} output tokens, {U["reasoning"]} hidden reasoning tokens)')
log('stub used: ' + ('YES, results are not real' if llm.stub_used() else 'no, every answer came from the API'))
save('ch4_parallel', {'model': llm.MODEL, 'stub': llm.stub_used(), 'n': N, 'vote_summary': vote_summary, 'vote_rows': vote_rows, 'per_sample_pass': per_sample,
                      'p24_summary': p24_summary, 'p24_rows': p24_rows, 'latency': lat, 'latency_mean': {'sequential_s': ms, 'concurrent_s': mp}, 'usage': U})
