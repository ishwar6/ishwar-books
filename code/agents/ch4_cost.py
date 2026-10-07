"""Chapter 4: the cost, latency and reliability arithmetic of the five workflow patterns. Pure arithmetic, no API.
Planning numbers are the same as Chapter 3's ch3_cost.py (a 1,500-token prompt, 300-token outputs, 2 s per call plus 0.02 s per output
token) so the two chapters' numbers can be compared. Five parts:
 1. the chain-reliability curve: P(chain succeeds) = r^n for per-step reliability r and n steps, and what a gate with one retry buys;
 2. routing: cost and accuracy as a function of the share of hard inputs and the router's two error rates;
 3. parallelisation: sectioning (latency = the slowest section) and voting (cost = n, accuracy from the binomial majority, if independent);
 4. orchestrator-workers: calls, tokens and latency against one agent doing the same k subtasks in sequence;
 5. evaluator-optimiser: expected rounds and cost per accepted answer with a reliable critic, and the precision of the stop with an unreliable one.
Results go to results/ch4_cost.json for the figures."""
import math
from common import Log, save

log = Log('ch4_cost')

# ---- planning numbers (tokens and seconds), round and deliberately ordinary
P = 1500        # prompt per call: instructions, schema or tool definitions, the input
OUT = 300       # a typical step output (a JSON object, a draft, a classification with a sentence)
BRIEF = 400     # what a chain step or a worker receives from the previous step or the orchestrator (not the whole history)
L_CALL = 2.0    # seconds per model call (fixed part)
L_TOK = 0.02    # seconds per output token
PRICE_SMALL, PRICE_FRONTIER = (0.25, 2.00), (3.00, 15.00)     # USD per million tokens (input, output)


def cost(inp, out, price=PRICE_FRONTIER):
    return inp / 1e6 * price[0] + out / 1e6 * price[1]


def lat(out_tokens, calls=1):
    return calls * L_CALL + L_TOK * out_tokens


one_call = dict(calls=1, inp=P, out=OUT, lat=lat(OUT), cost=cost(P, OUT))

# ---- 1. the chain-reliability curve
log('== 1. A chain of n steps: P(success) = r^n, and what a gate with one retry buys ==')
log('r = per-step reliability (the step returns a usable, correct output). A gate detects a share d of bad outputs and retries once:')
log("r' = r + (1 - r) * d * r. Undetected bad outputs (share (1 - r)(1 - d)) flow downstream and the chain fails.")
log('ASSUMES: steps fail independently; one bad step ruins the result (no later step repairs it); the retry succeeds with the same r as')
log('the first attempt (a feedback-driven retry may do better or worse); the gate never rejects a good output.')
chain = []
for r in [0.90, 0.95, 0.99]:
    for d in [0.0, 0.8]:
        rg = r + (1 - r) * d * r
        for n in range(1, 11):
            chain.append({'r': r, 'd': d, 'n': n, 'p_success': rg ** n, 'expected_calls': n * (1 + (1 - r) * d)})
log(f'{"r":>5} {"gate d":>7} ' + ' '.join(f'{"n=" + str(n):>6}' for n in [1, 2, 3, 5, 8, 10]) + '   expected calls at n=5')
for r in [0.90, 0.95, 0.99]:
    for d in [0.0, 0.8]:
        row = [c for c in chain if c['r'] == r and c['d'] == d]
        pick = {c['n']: c for c in row}
        log(f'{r:>5.2f} {d:>7.1f} ' + ' '.join(f'{pick[n]["p_success"]:>6.0%}' for n in [1, 2, 3, 5, 8, 10]) + f'   {pick[5]["expected_calls"]:.2f}')
log('reading: five steps at 90% each succeed 59% of the time; a gate that catches 80% of bad outputs and retries once lifts that to 87% for 8% more calls.')
log('the gate works only on failures it can see (a missing key, a bad enum, a failed regex); a plausible wrong value passes every schema check.')

# ---- 2. routing
log('')
log('== 2. Routing: cost and accuracy against the share of hard inputs and the router\'s error rates ==')
c_cheap, c_strong, c_router = cost(P, OUT, PRICE_SMALL), cost(P, OUT, PRICE_FRONTIER), cost(P, 5, PRICE_SMALL)
a = {'cheap_easy': 0.95, 'cheap_hard': 0.55, 'strong_easy': 0.97, 'strong_hard': 0.90}        # illustrative accuracies by model and input kind
log(f'one call: cheap ${c_cheap:.4f}, strong ${c_strong:.4f}, router ${c_router:.4f}; accuracies (illustrative): {a}')
log('ASSUMES: the router\'s two error rates do not depend on the input; accuracy of each model on each kind of input is fixed;')
log('one call per question; the accuracies are illustrative, not measured (our measured router is in ch4_router.py).')
log(f'{"hard share":>10} {"recall on hard":>14} {"false escal.":>12} {"accuracy":>9} {"cost $":>8} {"vs all-strong":>13} {"vs all-cheap acc":>16}')
routing = []
for h in [0.1, 0.3, 0.5]:
    for s, f in [(1.0, 0.0), (0.9, 0.1), (0.7, 0.2), (0.5, 0.5)]:
        acc = (1 - h) * ((1 - f) * a['cheap_easy'] + f * a['strong_easy']) + h * (s * a['strong_hard'] + (1 - s) * a['cheap_hard'])
        strong_share = (1 - h) * f + h * s
        c = c_router + (1 - strong_share) * c_cheap + strong_share * c_strong
        all_strong_acc = (1 - h) * a['strong_easy'] + h * a['strong_hard']
        all_cheap_acc = (1 - h) * a['cheap_easy'] + h * a['cheap_hard']
        routing.append({'hard_share': h, 'recall_hard': s, 'false_escalation': f, 'accuracy': acc, 'cost': c, 'strong_share': strong_share,
                        'all_strong_acc': all_strong_acc, 'all_strong_cost': c_strong, 'all_cheap_acc': all_cheap_acc, 'all_cheap_cost': c_cheap})
        log(f'{h:>10.0%} {s:>14.0%} {f:>12.0%} {acc:>9.1%} {c:>8.4f} {c / c_strong:>12.0%} {all_cheap_acc:>15.1%}')
log('reading: a perfect router (recall 100%, false escalation 0%) costs the strong price only on the hard share; every point of recall lost on hard')
log('inputs costs accuracy, every point of false escalation costs money. The confusion matrix of the router IS its cost model.')

# ---- 3. parallelisation
log('')
log('== 3. Parallelisation: sectioning and voting ==')
sect = []
for k in [1, 2, 4, 8]:
    seq_lat, par_lat = k * lat(OUT) + lat(OUT), lat(OUT) + lat(OUT)                 # k sections then an aggregation call
    sect.append({'sections': k, 'sequential_lat': seq_lat, 'parallel_lat': par_lat, 'cost': (k + 1) * cost(P, OUT)})
    log(f'  {k} sections: sequential {seq_lat:>5.1f} s, parallel {par_lat:>4.1f} s ({seq_lat / par_lat:.1f}x), cost {(k + 1) * cost(P, OUT):.4f} either way')
log('  latency of a fan-out is the slowest branch plus the aggregation; cost is the sum of the branches. Sectioning buys time, never tokens.')


def plurality(p, n, m):
    """P(the most frequent answer among n independent samples is the right one) when each sample is right with probability p and
    otherwise picks one of m distinct wrong answers uniformly ((1 - p) / m each). Ties are broken uniformly at random among the tied.
    Exact enumeration of the multinomial; fine for n <= 11 and m <= 5."""
    from itertools import product
    from math import factorial
    q = (1 - p) / m
    total = 0.0
    def comps(k, parts):                        # all ways to split k wrong samples over `parts` wrong answers
        if parts == 1:
            yield (k,)
            return
        for i in range(k + 1):
            for rest in comps(k - i, parts - 1):
                yield (i,) + rest
    for c in range(n + 1):
        for w in comps(n - c, m):
            prob = factorial(n) / (factorial(c) * math.prod(factorial(x) for x in w)) * p ** c * math.prod(q ** x for x in w)
            top = max((c,) + w)
            if c == top:
                total += prob / sum(1 for x in (c,) + w if x == top)
    return total


log('  plurality voting (the most frequent answer wins; ties broken at random). ASSUMES independent samples, a constant chance p of the')
log('  right answer, and wrong answers spread evenly over m distinct values. m = 1 is the binary case.')
vote = []
log(f'  {"p":>5} {"m wrong":>8} ' + ' '.join(f'{"n=" + str(n):>6}' for n in [1, 3, 5, 7, 11]))
for p, m in [(0.4, 1), (0.4, 2), (0.4, 5), (0.6, 1), (0.6, 2), (0.8, 2)]:
    row = {n: plurality(p, n, m) for n in [1, 3, 5, 7, 11]}
    vote.append({'p': p, 'm': m, **{f'n{n}': v for n, v in row.items()}})
    log(f'  {p:>5.1f} {m:>8} ' + ' '.join(f'{row[n]:>6.0%}' for n in [1, 3, 5, 7, 11]))
log('  reading: a right answer given 40% of the time LOSES a binary vote but WINS a plurality against two wrong answers at 30% each,')
log('  and wins more often the more scattered the wrong answers are. Voting needs the right answer to be the most frequent one, not a majority.')
log('  CORRELATED errors: if with probability rho all n samples copy one shared draw (the same misreading of the problem), accuracy is')
log('  rho * p + (1 - rho) * plurality(p, n, m). With p = 0.4, m = 2, n = 5:')
corr = []
for rho in [0.0, 0.3, 0.6, 0.9]:
    v = rho * 0.4 + (1 - rho) * plurality(0.4, 5, 2)
    corr.append({'rho': rho, 'acc': v})
    log(f'    rho = {rho:.1f}: {v:.0%}')
log('  Cost is n times one sample; latency is one sample if the calls run concurrently.')

# ---- 4. orchestrator-workers
log('')
log('== 4. Orchestrator-workers against one agent doing k subtasks in sequence ==')
orch = []
for k in [2, 4, 8]:
    plan_out, synth_out = 250, 400
    single_inp = sum(P + i * (OUT + 150) for i in range(k + 1))                        # one agent, re-reading its history each step (Chapter 3)
    single_out, single_lat = (k + 1) * OUT, (k + 1) * lat(OUT)
    o_inp = P + k * (BRIEF + P) + (P + k * OUT)                                        # plan + k workers with fresh contexts + synthesis
    o_out, o_lat = plan_out + k * OUT + synth_out, lat(plan_out) + lat(OUT) + lat(synth_out)
    orch.append({'k': k, 'single_inp': single_inp, 'single_out': single_out, 'single_lat': single_lat, 'single_cost': cost(single_inp, single_out),
                 'orch_inp': o_inp, 'orch_out': o_out, 'orch_lat': o_lat, 'orch_cost': cost(o_inp, o_out), 'orch_calls': k + 2})
    log(f'  k={k}: single agent {single_inp:>7,} in / {single_out:>5,} out tokens, {single_lat:>5.1f} s, ${cost(single_inp, single_out):.4f}   '
        f'orchestrator {o_inp:>7,} in / {o_out:>5,} out, {o_lat:>4.1f} s, ${cost(o_inp, o_out):.4f}   ({k + 2} calls)')
log('  ASSUMES: the single agent replays its whole growing history each step (no compaction, no prompt caching, which would cut its')
log('  input cost); workers get a fixed brief, run one call each, and run concurrently; the orchestrator does not re-plan.')
log('  reading: with fresh worker contexts the orchestrator is not more expensive in tokens than a loop that re-reads its history, and it is faster')
log('  by the width of the fan-out. The multiplier that Anthropic reports (about 15x the tokens of a chat) comes from workers that each run a full')
log('  tool-using loop of their own, not from the orchestration itself. Cost follows the sum of the workers; latency follows the slowest.')

# ---- 5. evaluator-optimiser
log('')
log('== 5. Evaluator-optimiser: rounds, cost per accepted answer, and the precision of the stop ==')
c_round = cost(P + OUT, OUT) + cost(P + OUT, 60)                                      # one generation (reads the critique) + one evaluation
ev = []
log(f'one round (generate + evaluate) costs ${c_round:.4f} at frontier prices; p = chance a round passes a RELIABLE critic; k = max rounds')
log(f'{"p":>5} {"k":>3} {"P(pass by k)":>13} {"E[rounds]":>10} {"E[cost]":>8} {"$/accepted":>11}')
for p in [0.3, 0.5, 0.7]:
    for k in [1, 2, 3, 4]:
        ps = 1 - (1 - p) ** k
        er = (1 - (1 - p) ** k) / p
        ev.append({'p': p, 'k': k, 'p_pass': ps, 'e_rounds': er, 'e_cost': c_round * er, 'cost_per_accepted': c_round * er / ps})
        log(f'{p:>5.1f} {k:>3} {ps:>13.0%} {er:>10.2f} {c_round * er:>8.4f} {c_round * er / ps:>11.4f}')
log('reading: as in Chapter 3, the k cancels: cost per accepted answer is one round divided by p. Most of the gain arrives by the third round.')
log('')
log('ASSUMES: rounds are independent with the same pass chance p, each round costs the same, and the critic is PERFECT. A feedback-driven')
log('round usually has a higher p than the first (the feedback helps), so treat these as a floor on the benefit, not a forecast.')
log('')
log('an IMPERFECT critic. Let p = P(draft is good), r = P(critic accepts | good) and f = P(critic accepts | bad), the false-accept rate.')
log('By Bayes, the share of ACCEPTED drafts that are good (the precision of the stop) is  p r / (p r + (1 - p) f).')
log('The false-accept rate is NOT one minus precision: precision also depends on how many drafts were good to begin with.')
prec = []
log(f'{"p good":>7} {"r":>5} {"f":>5} {"P(accept)":>10} {"precision of the stop":>22}')
for p, r, f in [(0.9, 1.0, 0.2), (0.5, 1.0, 0.2), (0.3, 1.0, 0.2), (0.5, 0.9, 0.1), (0.5, 0.7, 0.05), (0.3, 0.9, 0.3)]:
    acc = p * r + (1 - p) * f
    pr = p * r / acc
    prec.append({'p': p, 'r': r, 'f': f, 'p_accept': acc, 'precision': pr})
    log(f'{p:>7.1f} {r:>5.2f} {f:>5.2f} {acc:>10.0%} {pr:>22.1%}')
log('reading: with 90% good drafts, a critic that accepts 20% of bad ones still stops on a good draft 97.8% of the time; with 30% good drafts')
log('the same critic gives 68%. The harder the task, the more the critic\'s false-accept rate matters. ASSUMES r and f do not depend on the round.')

save('ch4_cost', {'params': dict(P=P, OUT=OUT, BRIEF=BRIEF, L_CALL=L_CALL, L_TOK=L_TOK, PRICE_SMALL=PRICE_SMALL, PRICE_FRONTIER=PRICE_FRONTIER),
                  'one_call': one_call, 'chain': chain, 'routing': routing, 'routing_unit': {'cheap': c_cheap, 'strong': c_strong, 'router': c_router, 'acc': a},
                  'sectioning': sect, 'voting': vote, 'voting_correlated': corr, 'orchestrator': orch, 'evaluator': ev, 'evaluator_round_cost': c_round, 'stop_precision': prec})
