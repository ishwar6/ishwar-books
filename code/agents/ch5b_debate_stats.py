"""Chapter 5, Section 5.4: statistics on the saved debate run (results/ch5b_debate.json). No API calls.
1. A paired sign test for debate against voting at equal calls: only problems where exactly one method is right carry information.
   Under "no difference", each such problem is a fair coin, so p = P(at least this lopsided) from the binomial distribution.
2. How the self-review control changed answers between rounds, next to the debate's changes (who talked themselves out of a right answer).
3. Conformity: how often an agent that was right at round 0 and in the minority switched to the majority's wrong answer."""
import json, math
from collections import Counter
from common import Log, save


def right(a, gold):                       # the same grading rule as ch5b_debate.py (one penny of tolerance)
    return a is not None and abs(float(a) - float(gold)) < 0.011


log = Log('ch5b_debate_stats')
D = json.load(open('results/ch5b_debate.json'))


def sign_p(a, b):
    """Two-sided exact sign test: a wins for method 1, b wins for method 2."""
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


out = {}
for mode, rows in D['rows'].items():
    log(f'== {mode.upper()} ==')
    o = {}
    for a, b, calls in [('debate1', 'vote6', 6), ('debate2', 'vote9', 9), ('debate1', 'review1', 6), ('debate2', 'review2', 9)]:
        wa = sum(r[a] and not r[b] for r in rows); wb = sum(r[b] and not r[a] for r in rows)
        p = sign_p(wa, wb)
        o[f'{a}_vs_{b}'] = {'wins_a': wa, 'wins_b': wb, 'p': p}
        log(f'  {a} vs {b} ({calls} calls each): {a} alone right on {wa}, {b} alone right on {wb}; two-sided sign test p = {p:.3f}')
    for arm, key in [('debate', 'rounds'), ('self-review', 'srounds')]:
        fl = Counter()
        for r in rows:
            for t in range(2):
                for i in range(3):
                    x, y = right(r[key][t][i], r['gold']), right(r[key][t + 1][i], r['gold'])
                    fl['wrong_to_right' if y and not x else 'right_to_wrong' if x and not y else 'kept_right' if x else 'kept_wrong'] += 1
        o[f'flips_{arm}'] = dict(fl)
        log(f'  {arm:<11} answer changes over 2 rounds x 3 agents x 30 problems: wrong->right {fl["wrong_to_right"]}, right->wrong {fl["right_to_wrong"]}, '
            f'kept right {fl["kept_right"]}, kept wrong {fl["kept_wrong"]}')
    # conformity: right but outvoted at round 0 (the other two agree on one wrong answer)
    minority, conformed = 0, 0
    for r in rows:
        r0, r1 = r['rounds'][0], r['rounds'][1]
        for i in range(3):
            others = [r0[j] for j in range(3) if j != i]
            if right(r0[i], r['gold']) and others[0] == others[1] and not right(others[0], r['gold']):
                minority += 1
                conformed += (r1[i] == others[0])
    o['minority_right'] = minority; o['minority_conformed'] = conformed
    log(f'  agents right at round 0 but outvoted by two agents agreeing on a wrong answer: {minority}; switched to that wrong answer in round 1: {conformed}')
    # the reverse: wrong but outvoted by two right agents
    wm, fixed = 0, 0
    for r in rows:
        r0, r1 = r['rounds'][0], r['rounds'][1]
        for i in range(3):
            others = [r0[j] for j in range(3) if j != i]
            if not right(r0[i], r['gold']) and all(right(x, r['gold']) for x in others):
                wm += 1
                fixed += right(r1[i], r['gold'])
    o['minority_wrong'] = wm; o['minority_wrong_fixed'] = fixed
    log(f'  agents wrong at round 0 while both others were right: {wm}; right in round 1: {fixed}')
    out[mode] = o
save('ch5b_debate_stats', out)
