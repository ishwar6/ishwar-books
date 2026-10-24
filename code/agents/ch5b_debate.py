"""Chapter 5, Section 5.4: debate against voting at the same number of model calls. Real API (gpt-5-mini, reasoning_effort=minimal).

Thirty multi-step word problems from Northwind's buying desk (data/ch5/b/questions.json, gold answers computed in Python).
For each problem and each of two working budgets we make 21 calls:
  * 9 independent samples (the "voters"). The first 3 double as round 0 of the debate.
  * debate round 1: each of the 3 agents sees the question, its own answer and the other two answers, and may revise (3 calls).
  * debate round 2: the same again on the round-1 answers (3 calls).
  * a CONTROL with the same shape and the same calls: each of the 3 agents re-checks only its OWN previous answer, for 2 rounds (6 calls).
    If the control gains as much as debate, the gain comes from a second look, not from reading the other agents.
So we can compare, call for call:
  single sample (1 call) | vote of 3 (3 calls) | vote of 6 (6) vs debate, 3 agents x 1 round (6) | vote of 9 (9) vs debate, 2 rounds (9).
The final answer of a vote or a debate is the plurality (most frequent answer; ties go to the earliest).
Two working budgets: TERSE (at most one short sentence before the answer) and BRIEF (at most 40 words of key working). The debate prompt
is the round-1 update prompt of Du et al. (2023), shortened, with the same working budget as the voters, so the debaters get no extra room.
Usage: python ch5b_debate.py"""
import json, os, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from common import Log, save, HERE
import ch4_llm as llm

log = Log('ch5b_debate')
llm.set_log(log)
QS = json.load(open(os.path.join(HERE, 'data', 'ch5', 'b', 'questions.json')))
BUDGET = {'terse': 'Do not show working; write at most one short sentence (under 20 words)',
          'brief': 'Show only key working, at most 40 words'}
SOLVE = 'Solve the problem. {budget}, then end with a line "Final answer: <number>".\n\nProblem: {q}'
DEBATE = ('Solve the problem. {budget}, then end with a line "Final answer: <number>".\n\nProblem: {q}\n\n'
          'Your previous answer was:\n{mine}\n\nThese are the answers from two other agents:\n{others}\n\n'
          'Use these opinions carefully as additional advice. Check the steps where you disagree, then give your updated answer.')
REVIEW = ('Solve the problem. {budget}, then end with a line "Final answer: <number>".\n\nProblem: {q}\n\n'
          'Your previous answer was:\n{mine}\n\nCheck your previous answer carefully step by step, then give your updated answer.')


def answer(text):
    """The number after the last 'Final answer:', else the last number in the text; returned as a canonical string."""
    m = re.findall(r'Final answer:\s*\**\s*[£$]?\s*(-?[\d,]*\.?\d+)', text or '')
    if not m:
        m = re.findall(r'-?\d[\d,]*\.?\d*', text or '')
    if not m:
        return None
    v = float(m[-1].replace(',', ''))
    return f'{v:.2f}'


def right(a, gold):
    return a is not None and abs(float(a) - float(gold)) < 0.011       # one penny of tolerance for money answers


def plurality(answers):
    c = Counter(a for a in answers if a is not None)
    if not c:
        return None
    best = max(c.values())
    return next(a for a in answers if a is not None and c[a] == best)   # ties go to the earliest answer


def run_question(q, mode):
    b = BUDGET[mode]
    use = {'vote': [], 'debate': []}
    samples, texts = [], []
    for i in range(9):
        t, u = llm.ask(SOLVE.format(budget=b, q=q['q']), max_tokens=900)
        texts.append(t); samples.append(answer(t))
        use['vote'].append(u)
    rounds, rtexts = [samples[:3]], [texts[:3]]
    for r in range(2):
        new, newt = [], []
        for i in range(3):
            others = '\n'.join(f'Agent {j + 1}: {rtexts[-1][j]}' for j in range(3) if j != i)
            t, u = llm.ask(DEBATE.format(budget=b, q=q['q'], mine=rtexts[-1][i], others=others), max_tokens=900)
            new.append(answer(t)); newt.append(t)
            use['debate'].append(u)
        rounds.append(new); rtexts.append(newt)
    srounds, stexts = [samples[:3]], [texts[:3]]                         # control: same agents, same calls, but each sees only its own answer
    for r in range(2):
        new, newt = [], []
        for i in range(3):
            t, u = llm.ask(REVIEW.format(budget=b, q=q['q'], mine=stexts[-1][i]), max_tokens=900)
            new.append(answer(t)); newt.append(t)
            use.setdefault('review', []).append(u)
        srounds.append(new); stexts.append(newt)
    g = q['gold']
    row = {'srounds': srounds, 'review1': right(plurality(srounds[1]), g), 'review2': right(plurality(srounds[2]), g),
           'out_vote': sum(u['output'] for u in use['vote']) / 9, 'out_debate': sum(u['output'] for u in use['debate']) / 6,
           'out_review': sum(u['output'] for u in use['review']) / 6,'id': q['id'], 'gold': g, 'mode': mode, 'samples': samples, 'rounds': rounds,
           'single': sum(right(a, g) for a in samples[:3]) / 3,                 # expected accuracy of one call
           'vote3': right(plurality(samples[:3]), g), 'vote6': right(plurality(samples[:6]), g), 'vote9': right(plurality(samples), g),
           'debate1': right(plurality(rounds[1]), g), 'debate2': right(plurality(rounds[2]), g),
           'tok_vote6': sum(u['input'] + u['output'] for u in use['vote'][:6]),
           'tok_debate1': sum(u['input'] + u['output'] for u in use['vote'][:3] + use['debate'][:3]),
           'cost_vote6': sum(llm.cost(u) for u in use['vote'][:6]), 'cost_debate1': sum(llm.cost(u) for u in use['vote'][:3] + use['debate'][:3]),
           'cost_vote9': sum(llm.cost(u) for u in use['vote']), 'cost_debate2': sum(llm.cost(u) for u in use['vote'][:3] + use['debate'])}
    # how individual agents changed their minds between rounds
    fl = Counter()
    for r in range(2):
        for i in range(3):
            a, b2 = right(rounds[r][i], g), right(rounds[r + 1][i], g)
            fl['right_to_wrong' if a and not b2 else 'wrong_to_right' if b2 and not a else 'kept_right' if a else 'kept_wrong'] += 1
    row['flips'] = dict(fl)
    row['unanimous0'] = len(set(rounds[0])) == 1
    row['unanimous2'] = len(set(rounds[2])) == 1
    row['converged_wrong'] = row['unanimous2'] and not right(rounds[2][0], g)
    row['lone_right_lost'] = sum(right(a, g) for a in rounds[0]) == 1 and not right(plurality(rounds[2]), g)
    return row


before = llm.snapshot()
log(f'== Debate against voting at equal calls: {len(QS)} problems x 2 working budgets, {llm.MODEL}, reasoning_effort=minimal ==')
log('   per problem: 9 independent samples; a 3-agent debate whose round 0 is the first 3 samples, then 2 rounds of 3 calls')
ALL = {}
for mode in ['terse', 'brief']:
    with ThreadPoolExecutor(max_workers=10) as ex:
        rows = list(ex.map(lambda q: run_question(q, mode), QS))
    ALL[mode] = rows
    log(f'\n-- budget: {mode.upper()} ("{BUDGET[mode]}") --')
    log(' id   gold      samples (9)                                    | debate r0 -> r1 -> r2')
    for r in rows:
        f = lambda xs: ' '.join((x or '-').rstrip('0').rstrip('.') for x in xs)
        mark = lambda ok: 'y' if ok else 'N'
        log(f' {r["id"]} {r["gold"]:>8}  {f(r["samples"])[:46]:<46} | {f(r["rounds"][0])} -> {f(r["rounds"][1])} -> {f(r["rounds"][2])}'
            f'  v6 {mark(r["vote6"])} d1 {mark(r["debate1"])} s1 {mark(r["review1"])} v9 {mark(r["vote9"])} d2 {mark(r["debate2"])} s2 {mark(r["review2"])}')

summary = {}
log('\n== Summary (accuracy over 30 problems; calls per problem in brackets) ==')
log(f'{"budget":<7}{"single(1)":>10}{"vote3(3)":>10}{"vote6(6)":>10}{"debate1(6)":>12}{"review1(6)":>12}{"vote9(9)":>10}{"debate2(9)":>12}{"review2(9)":>12}')
for mode, rows in ALL.items():
    n = len(rows)
    s = {k: sum(r[k] for r in rows) / n for k in ['single', 'vote3', 'vote6', 'debate1', 'review1', 'vote9', 'debate2', 'review2', 'out_vote', 'out_debate', 'out_review']}
    s.update({k: sum(r[k] for r in rows) for k in ['tok_vote6', 'tok_debate1']})
    s.update({k: sum(r[k] for r in rows) for k in ['cost_vote6', 'cost_debate1', 'cost_vote9', 'cost_debate2']})
    fl = Counter()
    for r in rows:
        fl.update(r['flips'])
    s['flips'] = dict(fl)
    s['unanimous0'] = sum(r['unanimous0'] for r in rows); s['unanimous2'] = sum(r['unanimous2'] for r in rows)
    s['converged_wrong'] = sum(r['converged_wrong'] for r in rows); s['lone_right_lost'] = sum(r['lone_right_lost'] for r in rows)
    s['lone_right'] = sum(sum(right(a, r['gold']) for a in r['rounds'][0]) == 1 for r in rows)
    s['debate1_wins'] = sum(r['debate1'] and not r['vote6'] for r in rows); s['vote6_wins'] = sum(r['vote6'] and not r['debate1'] for r in rows)
    s['debate2_wins'] = sum(r['debate2'] and not r['vote9'] for r in rows); s['vote9_wins'] = sum(r['vote9'] and not r['debate2'] for r in rows)
    summary[mode] = s
    log(f'{mode:<7}{s["single"]:>10.0%}{s["vote3"]:>10.0%}{s["vote6"]:>10.0%}{s["debate1"]:>12.0%}{s["review1"]:>12.0%}{s["vote9"]:>10.0%}{s["debate2"]:>12.0%}{s["review2"]:>12.0%}')
for mode, s in summary.items():
    log(f'\n{mode}: tokens for 6 calls, vote {s["tok_vote6"]:,} against debate {s["tok_debate1"]:,} ({s["tok_debate1"] / s["tok_vote6"]:.2f}x); '
        f'cost vote6 ${s["cost_vote6"]:.4f}, debate1 ${s["cost_debate1"]:.4f}, vote9 ${s["cost_vote9"]:.4f}, debate2 ${s["cost_debate2"]:.4f}')
    log(f'  mean output tokens per call: voter {s["out_vote"]:.0f}, debater {s["out_debate"]:.0f}, self-reviewer {s["out_review"]:.0f}')
    log(f'  problems where debate beat the vote at equal calls: 6 calls {s["debate1_wins"]}, 9 calls {s["debate2_wins"]}; '
        f'where the vote beat debate: {s["vote6_wins"]}, {s["vote9_wins"]}')
    f = s['flips']
    log(f'  agent answers across both debate rounds (180 = 30 problems x 3 agents x 2 rounds): wrong->right {f.get("wrong_to_right", 0)}, '
        f'right->wrong {f.get("right_to_wrong", 0)}, kept right {f.get("kept_right", 0)}, kept wrong {f.get("kept_wrong", 0)}')
    log(f'  unanimous at round 0: {s["unanimous0"]}/30, after round 2: {s["unanimous2"]}/30; unanimous on a WRONG answer after round 2: {s["converged_wrong"]}/30')
    log(f'  problems where exactly one agent was right at round 0: {s["lone_right"]}; of those, the debate ended wrong: {s["lone_right_lost"]}')
spent = llm.diff(before, llm.snapshot())
log(f'\nAPI: {spent["calls"]} calls, {spent["input"]:,} input and {spent["output"]:,} output tokens ({spent["reasoning"]:,} hidden reasoning), '
    f'${llm.cost(spent):.3f}. Stub used: {llm.stub_used()}')
save('ch5b_debate', {'summary': summary, 'rows': ALL, 'usage': spent, 'cost': llm.cost(spent), 'model': llm.MODEL})
