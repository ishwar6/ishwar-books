"""Chapter 3: the cost and latency arithmetic of the reasoning patterns. Pure arithmetic, no API.
For one task, with a fixed prompt size, a fixed thought size and a fixed observation size, count the model calls, the input and output
tokens and the wall-clock time of: direct answer, chain of thought, ReAct (n steps), plan-and-execute (1 planner + n executors, run in
parallel), tree of thoughts (breadth b, depth d, with a model-scored evaluation of every candidate), and best-of-n with a verifier.
Then the formula that decides between patterns: expected cost per correct answer = cost per attempt / accuracy, and its version with a
verifier and up to k retries. The planning numbers are stated at the top; change them for your own task."""
import json, math, os
from common import Log, save

log = Log('ch3_cost')

# ---- planning numbers (tokens), round and deliberately ordinary
P = 1500        # prompt per call: instructions + tool definitions + the question
THOUGHT = 150   # tokens of visible reasoning the model writes per step
ANSWER = 30     # tokens of the final answer
OBS = 300       # tokens a tool returns per call
COT_OUT = 350   # a chain-of-thought answer: several sentences plus the answer
PLAN = 250      # a written plan (plan-and-execute)
EXEC_IN = 600   # an executor call reads a short brief + one step, not the whole history
EXEC_OUT = 80
EVAL_OUT = 20   # a tree-of-thoughts evaluation ("sure / maybe / impossible" plus a sentence)
L_CALL = 2.0    # seconds per model call (fixed part)
L_TOK = 0.02    # seconds per output token (streamed generation)
L_TOOL = 0.5    # seconds per tool call
PRICES = {'small model ($0.25 / $2 per M)': (0.25, 2.00), 'frontier model ($3 / $15 per M)': (3.00, 15.00)}


def cost(inp, out, price):
    pi, po = price
    return inp / 1e6 * pi + out / 1e6 * po


def pattern_table(n_steps=5, b=3, d=3, n_best=5):
    rows = []
    # direct: one call
    rows.append(dict(pattern='direct answer', calls=1, inp=P, out=ANSWER, tools=0, lat=L_CALL + L_TOK * ANSWER))
    # chain of thought: one call, longer output
    rows.append(dict(pattern='chain of thought', calls=1, inp=P, out=COT_OUT, tools=0, lat=L_CALL + L_TOK * COT_OUT))
    # ReAct: n steps; each step re-reads everything so far (prompt + k thoughts + k observations)
    inp = sum(P + k * (THOUGHT + OBS) for k in range(n_steps)) + (P + n_steps * (THOUGHT + OBS))   # n tool steps + 1 finishing call
    out = (n_steps + 1) * THOUGHT
    rows.append(dict(pattern=f'ReAct, {n_steps} tool steps', calls=n_steps + 1, inp=inp, out=out, tools=n_steps,
                     lat=(n_steps + 1) * (L_CALL + L_TOK * THOUGHT) + n_steps * L_TOOL))
    # plan-and-execute: one planner call, n executor calls with small contexts (in parallel), one call to write the answer
    inp = P + n_steps * (EXEC_IN + OBS) + (P + PLAN + n_steps * EXEC_OUT)
    out = PLAN + n_steps * EXEC_OUT + THOUGHT
    rows.append(dict(pattern=f'plan-and-execute, {n_steps} steps (parallel)', calls=n_steps + 2, inp=inp, out=out, tools=n_steps,
                     lat=(L_CALL + L_TOK * PLAN) + (L_CALL + L_TOK * EXEC_OUT + L_TOOL) + (L_CALL + L_TOK * THOUGHT)))
    # tree of thoughts, breadth-first: at each level, b kept states each propose b children (b calls), every child is evaluated (b*b calls)
    gen_calls, eval_calls = b * d, b * b * d
    inp = gen_calls * (P + d * THOUGHT / 2) + eval_calls * (P + d * THOUGHT / 2)
    out = gen_calls * b * THOUGHT + eval_calls * EVAL_OUT
    rows.append(dict(pattern=f'tree of thoughts, b={b}, d={d}', calls=gen_calls + eval_calls, inp=round(inp), out=round(out), tools=0,
                     lat=d * ((L_CALL + L_TOK * b * THOUGHT) + (L_CALL + L_TOK * EVAL_OUT))))          # per level: a generate round then an evaluate round
    # best-of-n: n chain-of-thought samples in parallel, each scored by one verifier call (or by a free test)
    rows.append(dict(pattern=f'best-of-{n_best} with a model verifier', calls=2 * n_best, inp=n_best * P + n_best * (P + COT_OUT), out=n_best * COT_OUT + n_best * EVAL_OUT,
                     tools=0, lat=(L_CALL + L_TOK * COT_OUT) + (L_CALL + L_TOK * EVAL_OUT)))
    rows.append(dict(pattern=f'best-of-{n_best} with a free checker (tests)', calls=n_best, inp=n_best * P, out=n_best * COT_OUT, tools=0,
                     lat=L_CALL + L_TOK * COT_OUT))
    return rows


rows = pattern_table()
log('== 1. One task, six patterns: calls, tokens, latency and cost (planning numbers at the top of ch3_cost.py) ==')
log(f'prompt {P} tokens, thought {THOUGHT}, observation {OBS}, {L_CALL} s per call + {L_TOK} s per output token, {L_TOOL} s per tool call')
log('')
hdr = f'{"pattern":<42} {"calls":>5} {"input":>8} {"output":>7} {"latency":>8} {"small $":>9} {"frontier $":>11}'
log(hdr)
small, frontier = PRICES['small model ($0.25 / $2 per M)'], PRICES['frontier model ($3 / $15 per M)']
for r in rows:
    r['cost_small'] = cost(r['inp'], r['out'], small)
    r['cost_frontier'] = cost(r['inp'], r['out'], frontier)
    log(f'{r["pattern"]:<42} {r["calls"]:>5} {r["inp"]:>8,} {r["out"]:>7,} {r["lat"]:>7.1f}s {r["cost_small"]:>9.4f} {r["cost_frontier"]:>11.4f}')
base = rows[0]
log('')
log('relative to the direct answer (frontier price):')
for r in rows:
    log(f'  {r["pattern"]:<42} {r["cost_frontier"] / base["cost_frontier"]:>6.1f}x cost  {r["lat"] / base["lat"]:>5.1f}x latency')

# ---- 2. expected cost per correct answer
log('')
log('== 2. Expected cost per correct answer = cost per attempt / accuracy ==')
log('accuracies below are ILLUSTRATIVE except the three marked "mea" (measured in ch3_react.py on twelve questions; the plain ReAct loop,')
log('not the guarded one, which reached 100% there). A cost per correct answer with zero correct answers is undefined.')
measured = {}
try:
    R = json.load(open(os.path.join(os.path.dirname(__file__), 'results', 'ch3_react.json')))
    if not R.get('stub'):
        measured = {'direct answer': R['summary']['direct']['acc'], 'chain of thought': R['summary']['cot']['acc'], 'ReAct, 5 tool steps': R['summary']['react']['acc']}
except FileNotFoundError:
    pass
acc_illustrative = {'direct answer': 0.50, 'chain of thought': 0.60, 'ReAct, 5 tool steps': 0.85, 'plan-and-execute, 5 steps (parallel)': 0.80,
                    'tree of thoughts, b=3, d=3': 0.90, 'best-of-5 with a model verifier': 0.80, 'best-of-5 with a free checker (tests)': 0.90}
log(f'{"pattern":<42} {"accuracy":>9} {"$ / attempt":>12} {"$ / correct":>12}')
cpc = {}
for r in rows:
    a = measured.get(r['pattern'], acc_illustrative[r['pattern']])
    tag = 'measured' if r['pattern'] in measured else 'assumed'
    c = r['cost_frontier']
    cpc[r['pattern']] = (c / a) if a else None
    log(f'{r["pattern"]:<42} {a:>6.0%} {tag[:3]} {c:>12.4f} ' + (f'{c / a:>12.4f}' if a else '   undefined (no correct answers)'))
log('')
log('reading: a pattern that costs 10x more per attempt is still cheaper per correct answer only if it is also more than 10x as accurate,')
log('or if a wrong answer has a cost of its own (a human review, a refund, a retry) that you add to the numerator.')

# ---- 3. a verifier and retries: how accuracy compounds when you can tell right from wrong
log('')
log('== 3. With a verifier (tests, a checker, a human), retry up to k times ==')
log('P(success within k) = 1 - (1 - p)^k      expected cost = c * (1 - (1 - p)^k) / p      (c = cost of one attempt, p = pass rate per attempt)')
c = rows[1]['cost_frontier']      # one chain-of-thought attempt at the frontier price
log(f'one attempt c = ${c:.4f} (chain of thought, frontier price)')
log(f'{"p per attempt":>14} {"k":>3} {"P(success)":>11} {"E[attempts]":>12} {"E[cost]":>9} {"$/correct":>10}')
retry = []
for p in [0.3, 0.5, 0.7]:
    for k in [1, 2, 3, 5]:
        ps = 1 - (1 - p) ** k
        ea = (1 - (1 - p) ** k) / p
        retry.append({'p': p, 'k': k, 'p_success': ps, 'e_attempts': ea, 'e_cost': c * ea, 'cost_per_correct': c * ea / ps})
        log(f'{p:>14.0%} {k:>3} {ps:>11.0%} {ea:>12.2f} {c * ea:>9.4f} {c * ea / ps:>10.4f}')
log('reading: with a verifier, retries raise the success rate quickly (p=0.5, k=3 gives 88%) and the cost per correct answer barely moves,')
log('because failed attempts are cheap and recognised. Without a verifier there is nothing to retry on: you pay k times and keep the same accuracy.')

# ---- 4. what the ReAct history costs as steps grow (why plan-and-execute separates the contexts)
log('')
log('== 4. ReAct input tokens against step count (each step re-reads the whole history) ==')
growth = []
for n in [2, 5, 10, 20]:
    inp = sum(P + k * (THOUGHT + OBS) for k in range(n + 1))
    pe = P + n * (EXEC_IN + OBS) + (P + PLAN + n * EXEC_OUT)
    growth.append({'steps': n, 'react_input': inp, 'plan_exec_input': pe})
    log(f'  {n:>2} steps: ReAct {inp:>8,} input tokens   plan-and-execute {pe:>7,}   ratio {inp / pe:.1f}x')

save('ch3_cost', {'params': dict(P=P, THOUGHT=THOUGHT, ANSWER=ANSWER, OBS=OBS, COT_OUT=COT_OUT, PLAN=PLAN, EXEC_IN=EXEC_IN, EXEC_OUT=EXEC_OUT,
                                 EVAL_OUT=EVAL_OUT, L_CALL=L_CALL, L_TOK=L_TOK, L_TOOL=L_TOOL), 'prices': PRICES, 'rows': rows,
                  'measured_acc': measured, 'assumed_acc': acc_illustrative, 'cost_per_correct': cpc, 'retry': retry, 'growth': growth})
