"""Chapter 4, Section 4.3: routing. Twenty-four short questions with exact answers in three designed tiers (easy, medium, hard) are
answered four ways: always-cheap, always-strong, a cheap-first cascade (two cheap samples; if they agree, accept; if not, escalate to
strong), and a classifier router (one cheap call labels the question easy or hard, then dispatches). "Cheap" is gpt-5-mini with
reasoning_effort=minimal; "strong" is THE SAME MODEL with reasoning_effort=medium, so that the whole experiment sits on one verified
price pair. That is a real cost difference (the medium setting spends hidden reasoning tokens, billed as output) but a smaller quality
gap than two different models would give; the chapter says so. Reports accuracy, cost, cost per correct answer, the cascade's
escalation rate and two confusion matrices. Grading is typed and exact (ch4_grade.py) for the router: against the designed tier and against the oracle (did cheap get it right?)."""
import re
from common import Log, save
import ch4_llm as llm

log = Log('ch4_router')
llm.set_log(log)
CHEAP, STRONG = 'minimal', 'medium'

# (question, gold answer, designed tier). Answers are compared after normalising case, punctuation and number formatting.
TASKS = [
    ('What is the capital of Portugal?', 'lisbon', 'easy'),
    ('Is the sentiment of this review positive or negative? "The battery died in a week and support never replied."', 'negative', 'easy'),
    ('Which of these is a mammal: trout, sparrow, dolphin, gecko?', 'dolphin', 'easy'),
    ('How many days are there in a leap year?', '366', 'easy'),
    ('Spell the word "necessary" backwards.', 'yrassecen', 'easy'),
    ('What is 15% of 240?', '36', 'easy'),
    ('Which language is this sentence in: "Onde fica a estação de comboios?"', 'portuguese', 'easy'),
    ('What is the plural of "criterion"?', 'criteria', 'easy'),
    ('A shop sells 3 pens for 2.40 and notebooks at 1.75 each. Dana buys 9 pens and 4 notebooks. What is the total cost?', '14.20', 'medium'),
    ('A rectangle is 3 longer than twice its width and its perimeter is 48. What is its area?', '119', 'medium'),
    ('Prices rose 20% and then fell 25%. A jacket now costs 72. What was the original price?', '80', 'medium'),
    ('Three friends split a bill of 138: Ana pays twice what Ben pays and Cy pays 18 more than Ben. How much does Ana pay?', '60', 'medium'),
    ('A tank holds 350 litres and starts with 70. A pipe adds 12 litres a minute while a drain removes 5. After how many minutes is it full?', '40', 'medium'),
    ('The sum of three consecutive even numbers is 150. What is the product of the smallest and the largest?', '2496', 'medium'),
    ('A cyclist rides 36 km at 12 km/h and returns at 18 km/h. What is the average speed for the whole trip in km/h?', '14.4', 'medium'),
    ('What is the sum of all two-digit numbers divisible by 7?', '728', 'medium'),
    ('How many trailing zeros does 25 factorial have?', '6', 'hard'),
    ('What is the sum of the digits of 2 to the power 20?', '31', 'hard'),
    ('In how many ways can 5 people sit in a row if two particular people must sit next to each other?', '48', 'hard'),
    ('What is the remainder when 7 to the power 100 is divided by 5?', '1', 'hard'),
    ('What day of the week was 1 January 2000?', 'saturday', 'hard'),
    ('If today is Wednesday, what day of the week will it be 100 days from now?', 'friday', 'hard'),
    ('What is the smallest positive integer divisible by every integer from 1 to 10?', '2520', 'hard'),
    ('A clock shows 3:15. What is the angle in degrees between the hour and minute hands?', '7.5', 'hard'),
]
ANSWER_PROMPT = 'Answer the question. Reply with ONLY the final answer (a word, a name or a number), no explanation.\n\nQuestion: {q}'
ROUTER_PROMPT = ('You are a router in front of two models: a small fast one and a large careful one. Decide which should answer the question below. '
                 'Questions that need several steps of calculation, counting or careful logic should go to the large model. '
                 'Reply with exactly one word: "easy" (small model) or "hard" (large model).\n\nQuestion: {q}')


from ch4_grade import grade, numbers_in, _clean


def norm(s):
    """Canonical key used only to compare two samples with each other (cascade agreement), never to grade."""
    ns = numbers_in(s)
    return f'{ns[0]:g}' if len(ns) == 1 else re.sub(r'[^a-z0-9:\- ]', '', _clean(s))


def correct(ans, gold):
    return grade(ans, gold)                  # typed, exact: see ch4_grade.py and its adversarial self-test


def answer(q, effort):
    text, u = llm.ask(ANSWER_PROMPT.format(q=q), effort=effort, max_tokens=(400 if effort == CHEAP else 6000))
    return text, u


# ---- run everything once per question and reuse the samples across conditions (the cascade's first cheap sample is the always-cheap answer)
rows = []
log(f'== Routing: {len(TASKS)} questions, cheap = {llm.MODEL} ({CHEAP} effort), strong = the same model at {STRONG} effort ==')
log('conditions: always-cheap | always-strong | cascade (two cheap samples agree -> accept, else strong) | classifier router (cheap call: easy/hard)')
log('')
for q, gold, tier in TASKS:
    c1, u_c1 = answer(q, CHEAP)
    c2, u_c2 = answer(q, CHEAP)
    s, u_s = answer(q, STRONG)
    r, u_r = llm.ask(ROUTER_PROMPT.format(q=q), effort=CHEAP, max_tokens=20)
    route = 'hard' if 'hard' in (r or '').lower() else 'easy'
    agree = norm(c1) == norm(c2)
    rows.append({'q': q, 'gold': gold, 'tier': tier, 'cheap': c1, 'cheap2': c2, 'strong': s, 'route': route, 'agree': agree,
                 'cheap_ok': correct(c1, gold), 'cheap2_ok': correct(c2, gold), 'strong_ok': correct(s, gold),
                 'u_cheap': u_c1, 'u_cheap2': u_c2, 'u_strong': u_s, 'u_router': u_r})

log(f'{"#":>2} {"tier":<6} {"gold":<10} {"cheap":<14} {"ok":<3} {"strong":<14} {"ok":<3} {"agree":<5} {"route":<5}')
for i, r in enumerate(rows):
    log(f'{i + 1:>2} {r["tier"]:<6} {r["gold"]:<10} {norm(r["cheap"])[:14]:<14} {"y" if r["cheap_ok"] else "N":<3} {norm(r["strong"])[:14]:<14} '
        f'{"y" if r["strong_ok"] else "N":<3} {"yes" if r["agree"] else "no":<5} {r["route"]:<5}')


def tally(name, pick):
    """pick(row) -> (correct, usage_list). Sums accuracy and cost for a condition."""
    ok, tot = 0, {'input': 0, 'output': 0, 'reasoning': 0}
    calls = 0
    for r in rows:
        c, us = pick(r)
        ok += int(c); calls += len(us)
        for u in us:
            for k in tot:
                tot[k] += u[k]
    cost = llm.cost(tot)
    return {'condition': name, 'correct': ok, 'n': len(rows), 'acc': ok / len(rows), 'calls': calls, 'tokens': tot, 'cost_usd': cost,
            'cost_per_correct': cost / ok if ok else None}


def cascade(r):
    if r['agree']:
        return r['cheap_ok'], [r['u_cheap'], r['u_cheap2']]
    return r['strong_ok'], [r['u_cheap'], r['u_cheap2'], r['u_strong']]


def router(r):
    if r['route'] == 'easy':
        return r['cheap_ok'], [r['u_router'], r['u_cheap']]
    return r['strong_ok'], [r['u_router'], r['u_strong']]


summary = [tally('always-cheap', lambda r: (r['cheap_ok'], [r['u_cheap']])),
           tally('always-strong', lambda r: (r['strong_ok'], [r['u_strong']])),
           tally('cascade', cascade),
           tally('classifier router', router),
           tally('oracle router', lambda r: (r['cheap_ok'] or r['strong_ok'], [r['u_cheap']] if r['cheap_ok'] else [r['u_strong']]))]
log('')
log('-- summary (cost from the API usage fields; hidden reasoning tokens are billed as output) --')
log(f'{"condition":<18} {"acc":>7} {"calls":>5} {"in tok":>8} {"out tok":>8} {"$ per 1k q":>10} {"$ per 1k correct":>16} {"x cheap":>8}')
base = summary[0]['cost_usd']
for s in summary:
    log(f'{s["condition"]:<18} {s["correct"]:>2}/{s["n"]:<2} {s["acc"]:>3.0%} {s["calls"]:>5} {s["tokens"]["input"]:>8,} {s["tokens"]["output"]:>8,} {s["cost_usd"] / s["n"] * 1000:>10.3f} '
        + (f'{s["cost_per_correct"] * 1000:>16.3f}' if s['cost_per_correct'] else f'{"n/a":>16}') + f' {s["cost_usd"] / base:>7.1f}x')
esc = sum(1 for r in rows if not r['agree'])
log(f'cascade escalated {esc}/{len(rows)} questions ({esc / len(rows):.0%}); router sent {sum(1 for r in rows if r["route"] == "hard")}/{len(rows)} to strong')

# ---- the router's confusion matrices
by_tier = {t: {'easy': 0, 'hard': 0} for t in ['easy', 'medium', 'hard']}
for r in rows:
    by_tier[r['tier']][r['route']] += 1
log('')
log('-- router decision against the designed tier (rows: tier; columns: where the router sent it) --')
log(f'{"tier":<8} {"-> cheap":>9} {"-> strong":>10}')
for t, d in by_tier.items():
    log(f'{t:<8} {d["easy"]:>9} {d["hard"]:>10}')
oracle = {'cheap suffices': {'easy': 0, 'hard': 0}, 'needs strong': {'easy': 0, 'hard': 0}}
for r in rows:
    oracle['cheap suffices' if r['cheap_ok'] else 'needs strong'][r['route']] += 1
log('')
log('-- router decision against the oracle (rows: did the cheap model get it right?; columns: where the router sent it) --')
log(f'{"truth":<16} {"-> cheap":>9} {"-> strong":>10}')
for t, d in oracle.items():
    log(f'{t:<16} {d["easy"]:>9} {d["hard"]:>10}')
tp = oracle['needs strong']['hard']; fn = oracle['needs strong']['easy']; fp = oracle['cheap suffices']['hard']; tn = oracle['cheap suffices']['easy']
log(f'reading: {fn} question(s) that needed the strong model were sent to the cheap one (lost accuracy); {fp} that the cheap model could answer '
    f'were sent to the strong one (wasted cost). recall on hard = {tp / max(1, tp + fn):.0%}, false escalation = {fp / max(1, fp + tn):.0%}')

U = llm.snapshot()
log('')
log(f'total spend: ${llm.cost(U):.4f} ({U["calls"]} calls, {U["input"]:,} input + {U["output"]:,} output tokens, {U["reasoning"]:,} hidden reasoning tokens)')
log('stub used: ' + ('YES, results are not real' if llm.stub_used() else 'no, every answer came from the API'))
save('ch4_router', {'model': llm.MODEL, 'cheap_effort': CHEAP, 'strong_effort': STRONG, 'stub': llm.stub_used(), 'summary': summary, 'rows': rows,
                    'by_tier': by_tier, 'oracle': oracle, 'usage': U})
