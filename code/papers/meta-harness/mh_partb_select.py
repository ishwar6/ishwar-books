"""Background page, part 2: how optimizers choose what to try next. No model runs; it reads results/partb_traces.jsonl.
1. GEPA-style Pareto selection over our three harnesses: keep every candidate that is the best on at least one question
   and not dominated. Method 1 GEPA's pool rule; Method 2 the definition of Pareto dominance; check they agree.
2. The PUCT score that TTT-Discover uses to pick which earlier solution to build on: Q(s) + c * P(s) * sqrt(1 + T / (1 + n(s))).
3. An OPRO-style meta-prompt: past (prompt, score) pairs sorted from low to high, from our own three harnesses.
Writes results/partb_select_stdout.txt and results/partb_select.json."""
import json, math, os
from collections import defaultdict
from common import Log, save, RESULTS

log = Log('partb_select')
MODEL = 'Qwen2.5-1.5B-Instruct'
rows = [json.loads(l) for l in open(os.path.join(RESULTS, 'partb_traces.jsonl'))]
score = defaultdict(dict)                                         # score[harness][question] = 1 if right else 0
for r in rows:
    if r['model'] == MODEL:
        score[r['harness']][r['question']] = int(r['pred'] == r['gold'])
names = list(score)
questions = list(score[names[0]])
R = {}

log('$ python mh_partb_select.py')
log(f'== 1. Pareto selection over our harnesses ({MODEL}, {len(questions)} questions)')
for h in names:
    log(f'  {h:10} right on {sum(score[h].values()):2} of {len(questions)}')


def gepa_pool(score):
    """Method 1 (GEPA-style): for each question, collect the candidates with the best score on it;
    then drop any collected candidate that another collected candidate dominates."""
    pool = {h for q in questions for h in score if score[h][q] == max(score[o][q] for o in score)}
    return sorted(h for h in pool if not any(dominated(h, o) for o in pool if o != h))


def dominated(a, b):
    """b dominates a if b is at least as good on every question and better on at least one."""
    va, vb = [score[a][q] for q in questions], [score[b][q] for q in questions]
    return all(y >= x for x, y in zip(va, vb)) and any(y > x for x, y in zip(va, vb))


def pareto_front(score):
    """Method 2: the definition itself, the candidates that no other candidate dominates."""
    return sorted(h for h in score if not any(dominated(h, o) for o in score if o != h))


m1, m2 = gepa_pool(score), pareto_front(score)
unique = {h: [q for q in questions if score[h][q] == 1 and all(score[o][q] == 0 for o in names if o != h)]
          for h in names}
for h in names:
    log(f'  {h:10} the only harness that is right on {len(unique[h]):2} questions')
log(f'  Method 1 (GEPA-style pool):  {m1}')
log(f'  Method 2 (not dominated):    {m2}')
log(f'  same: {m1 == m2}')
for h in names:
    if unique[h]:
        log(f'  only {h} gets this one right: {unique[h][0]!r}')
R['pareto'] = dict(unique={h: len(v) for h, v in unique.items()}, method1=m1, method2=m2)


# ---------------------------------------------------------------- 2. PUCT
def puct(Q, P, n, T, c=1.0):
    """TTT-Discover's reuse rule: exploit a high reward Q, trust a high prior P, explore what was rarely expanded."""
    return Q + c * P * math.sqrt(1 + T / (1 + n))


log('')
log('== 2. The PUCT score: which earlier solution to build on next (c = 1, T = 8 expansions so far)')
cands = [('A: best reward, built on 7 times', 0.90, 0.6, 7), ('B: good reward, built on once', 0.80, 0.5, 1),
         ('C: weak reward, never built on', 0.40, 0.2, 0)]
R['puct'] = []
for name, Q, P, n in cands:
    s = puct(Q, P, n, T=8)
    log(f'  {name:34} Q={Q:.2f} P={P:.1f} n={n}   score = {Q:.2f} + {P:.1f} * sqrt(1 + 8/{1 + n}) = {s:.3f}')
    R['puct'].append(dict(name=name, Q=Q, P=P, n=n, score=s))
log(f'  picked: {max(R["puct"], key=lambda d: d["score"])["name"]}')

# ---------------------------------------------------------------- 3. an OPRO-style meta-prompt
acc = {h: 100 * sum(score[h].values()) / len(questions) for h in names}
desc = {'zero-shot': 'Show only the list of labels.', 'few-shot': 'Show one fixed example question for each label.',
        'retrieval': 'Show the 6 most similar labelled questions from memory.'}
log('')
log('== 3. An OPRO-style meta-prompt built from our three harnesses (sorted from low to high score)')
meta = ['Here are earlier instructions with their accuracy. Higher is better.', '']
for h in sorted(names, key=lambda h: acc[h]):
    meta += [f'Instruction: {desc[h]}', f'Accuracy: {acc[h]:.1f}', '']
meta.append('Write a new instruction that is different from all of the above and has a higher accuracy.')
for line in meta:
    log('  | ' + line)
R['meta_prompt'] = meta
save('partb_select', R)
