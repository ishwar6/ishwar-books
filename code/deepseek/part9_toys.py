"""Part 9: two toy re-implementations of DeepInfer scheduling ideas, on made-up numbers.

    ~/.venvs/attn/bin/python part9_toys.py

1. Variable-length DSpark verification: the greedy in csrc/engine/mtp_varlen.hpp (MTPVarlen::schedule),
   re-written in Python. The performance curve below is invented for the example, not measured.
2. EPLB: the two greedy steps of deepinfer/model_executor/eplb/policy.py (rebalance_experts_global and
   _batch_greedy_indices), re-written for one layer with plain Python lists. The expert loads are invented.
Nothing from DeepInfer is imported or run. Output: results/part9_toys.json and _stdout.txt.
"""
from common import Log, save

log = Log('part9_toys')

# ---------------------------------------------------------------- 1. variable-length verification
log('=== 1. Variable-length verification (toy) ===')
# confidence of each of 5 draft tokens, for 4 requests: conditional "survives if the previous one survived"
conf = {
    'A (code)': [0.95, 0.93, 0.90, 0.88, 0.85],
    'B (chat)': [0.80, 0.60, 0.50, 0.40, 0.30],
    'C (chat)': [0.70, 0.50, 0.40, 0.30, 0.20],
    'D (json)': [0.97, 0.95, 0.94, 0.92, 0.90],
}
# invented decode "performance curve": (tokens in the forward pass, milliseconds per step)
curve = [(4, 10.0), (8, 10.4), (12, 11.0), (16, 12.0), (20, 13.6), (24, 15.6)]


def step_ms(total):
    return next(ms for size, ms in curve if size >= total)   # smallest captured graph that fits


survive = {}
for r, cs in conf.items():
    a, p = [], 1.0
    for c in cs:
        p *= c
        a.append(p)          # survival of the whole prefix up to this draft token
    survive[r] = a
    log(f'  {r}: survival of each prefix = ' + ', '.join(f'{x:.2f}' for x in a))

queue = sorted(((a, r) for r, aa in survive.items() for a in aa), key=lambda t: -t[0])
R = len(conf)
expected, total = float(R), R                   # every request verifies its own real token first
best = (expected / step_ms(total), total, expected)
trace = [(total, expected, step_ms(total), expected / step_ms(total))]
for a, r in queue:
    total += 1
    expected += a
    if total > curve[-1][0]:
        break
    tp = expected / step_ms(total)
    trace.append((total, expected, step_ms(total), tp))
    if tp > best[0]:
        best = (tp, total, expected)
log('\n  tokens in pass | expected accepted | ms per step | expected tokens per ms')
for t, e, ms, tp in trace:
    mark = '  <- best' if t == best[1] else ''
    log(f'  {t:14d} | {e:17.2f} | {ms:11.1f} | {tp:8.3f}{mark}')
alloc = {r: 0 for r in conf}
for a, r in queue[:best[1] - R]:
    alloc[r] += 1
fixed_e = R + sum(sum(aa) for aa in survive.values())
fixed_tp = fixed_e / step_ms(R * 6)
log(f'\n  chosen: {best[1]} tokens, draft tokens verified per request = {alloc}')
log(f'  fixed-5 for everyone: {R * 6} tokens, {fixed_e:.2f} expected accepted, {fixed_tp:.3f} tokens/ms')
log(f'  varlen: {best[0]:.3f} tokens/ms, {100 * (best[0] / fixed_tp - 1):+.1f}% on this invented curve')

# ---------------------------------------------------------------- 2. EPLB
log('\n=== 2. Expert-parallel load balancing (toy, one layer) ===')
loads = [  # tokens routed to each of 8 logical experts, in 3 recent steps
    [90, 10, 10, 20, 15, 15, 30, 10],
    [80, 15, 10, 25, 10, 20, 35, 5],
    [100, 5, 15, 20, 10, 10, 30, 10],
]
n_log, n_gpu, n_redundant = 8, 4, 4
n_phy = n_log + n_redundant
cap = n_phy // n_gpu
steps = len(loads)


def rank_loads(assign, count):
    """assign: list of GPUs, each a list of logical experts (replicas share the load equally)."""
    return [[sum(loads[s][e] / count[e] for e in g) for g in assign] for s in range(steps)]


def score(rl):
    return sum(max(r) for r in rl) / sum(sum(r) / len(r) for r in rl)   # mean over steps of (max / mean)


naive = [[0, 1], [2, 3], [4, 5], [6, 7]]
rl0 = rank_loads(naive, [1] * n_log)
log(f'  no replicas, experts in order: per-step GPU loads {[[round(x) for x in r] for r in rl0]}, '
    f'slowest/mean = {score(rl0):.2f}')

# step 1: give each redundant slot to the expert whose per-replica load is highest (summed over steps)
count = [1] * n_log
phys = list(range(n_log))
for _ in range(n_redundant):
    best_e, best_v = None, None
    for e in range(n_log):
        trial = [c + (1 if k == e else 0) for k, c in enumerate(count)]
        v = sum(max(loads[s][k] / trial[k] for k in range(n_log)) for s in range(steps))
        if best_v is None or v < best_v:
            best_e, best_v = e, v
    count[best_e] += 1
    phys.append(best_e)
log(f'  replicas per expert after step 1: {count}')

# step 2: place replicas, heaviest first, on the GPU that keeps the sum over steps of the peak load lowest
per_rep = {e: sum(loads[s][e] for s in range(steps)) / count[e] for e in range(n_log)}
order = sorted(range(n_phy), key=lambda i: -per_rep[phys[i]])
gpu = [[] for _ in range(n_gpu)]
gl = [[0.0] * steps for _ in range(n_gpu)]
for i in order:
    e = phys[i]
    vec = [loads[s][e] / count[e] for s in range(steps)]
    best_g, best_v = None, None
    for g in range(n_gpu):
        if len(gpu[g]) == cap:
            continue
        v = sum(max([gl[g][s] + vec[s]] + [gl[h][s] for h in range(n_gpu) if h != g]) for s in range(steps))
        if best_v is None or v < best_v:
            best_g, best_v = g, v
    gpu[best_g].append(e)
    gl[best_g] = [gl[best_g][s] + vec[s] for s in range(steps)]
rl1 = rank_loads(gpu, count)
log(f'  placement (logical expert ids per GPU): {gpu}')
log(f'  per-step GPU loads {[[round(x) for x in r] for r in rl1]}, slowest/mean = {score(rl1):.2f}')

save('part9_toys', dict(varlen=dict(conf=conf, survive=survive, curve=curve, trace=trace, best_total=best[1],
                                    best_tp=best[0], alloc=alloc, fixed_tp=fixed_tp),
                        eplb=dict(loads=loads, naive=naive, naive_ratio=score(rl0), count=count, placement=gpu,
                                  ratio=score(rl1), rank_loads_before=rl0, rank_loads_after=rl1)))
