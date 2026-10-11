"""What breaks in production (simulated with engine.py, step costs fitted to this laptop's measurements).

1. Retry storms: clients give up after a timeout and retry. Below capacity nothing happens; past it,
   retries add load exactly when there is none to spare. Compared: no retries; immediate retries with a
   server that keeps working on abandoned requests; retries with exponential backoff and a server that cancels.
2. Preemption storms: a small KV cache. As load grows, running requests are evicted and recomputed.
3. Head-of-line blocking: one 16,000-token prompt arrives among short chats; with and without chunked prefill.
4. Noisy neighbour: one tenant sends a burst of long prompts; what happens to everyone else's first token.
Run: python code/serving/failures.py  -> results/failures.json, results/failures_stdout.txt
"""
import math, random
from engine import Engine, StallEngine, Request, poisson_workload, run, summarize, pct, lognormal_int
from common import Log, laptop_cost

log = Log('failures')
cost = laptop_cost()
OUT = {}

# ---------------------------------------------------------------- 1. retry storms after a short stall
TIMEOUT, RATE, T_STALL, STALL, T_END = 10_000.0, 13, 60_000, 10_000, 300_000
SHORT = (32, 0.3, 8, 64)                 # short answers, so a healthy server answers well inside the timeout
log(f'1. RETRY STORM: {RATE} requests/s of short answers (~32 tokens), client timeout {TIMEOUT / 1000:.0f} s, up to 3 retries.')
log(f'   The engine freezes for {STALL / 1000:.0f} s at t = {T_STALL // 1000} s. Success = the user got an answer within the timeout on some attempt.')
rows = {}
for mode in ('no retries', 'retry at once, no cancel', 'retry at once, cancel', 'backoff + jitter, cancel'):
    reqs = poisson_workload(RATE, int(RATE * T_END / 1000), seed=31, output=SHORT)
    rng = random.Random(7)
    for r in reqs: r.deadline = r.arrival + TIMEOUT
    def retry(r):
        if mode == 'no retries' or r.attempt >= 3: return None
        delay = 0.0 if mode.startswith('retry at once') else rng.uniform(0, 2000 * 2 ** r.attempt)
        t = r.deadline + delay
        return Request(-1, t, r.prompt, r.output, origin=r.origin, attempt=r.attempt + 1, deadline=t + TIMEOUT)
    e = StallEngine(cost, budget=2048, max_seqs=256, at=T_STALL, stall=STALL, cancel_on_timeout=mode.endswith(', cancel'))
    sub = run(reqs, [e], retry=retry)
    ok_at = {}
    for r in sub:
        if r.finish is not None and r.finish <= r.deadline:
            ok_at[r.origin] = min(ok_at.get(r.origin, math.inf), r.finish)
    win = []
    for w0 in range(0, T_END, 10_000):                 # per 10 s window of ORIGINAL arrival time
        orig = [r for r in reqs if w0 <= r.arrival < w0 + 10_000]
        att = [r for r in sub if w0 <= r.arrival < w0 + 10_000]
        win.append({'t': w0 / 1000, 'success': sum(r.origin in ok_at for r in orig) / max(1, len(orig)),
                    'attempts_per_s': len(att) / 10})
    after = [w for w in win if w['t'] * 1000 >= T_STALL + STALL]
    rec = next((w['t'] for w in after if w['success'] >= 0.95), None)
    rows[mode] = {'attempts': len(sub), 'success': len(ok_at) / len(reqs), 'recovered_at': rec, 'windows': win,
                  'wasted_tokens': sum(r.generated for r in sub if r.finish is not None and r.finish > r.deadline)}
    log(f'   {mode:26s}: attempts {len(sub):6d} for {len(reqs)} requests, success {len(ok_at) / len(reqs) * 100:5.1f}%, '
        f'back to >= 95% success for arrivals from t = {rec} s, tokens generated for users who had left: {rows[mode]["wasted_tokens"]:,}')
OUT['retry'] = rows

# ---------------------------------------------------------------- 2. preemption storms
KV = 100_000
log()
log(f'2. PREEMPTION: KV cache limited to {KV:,} tokens (all memory, on purpose); outputs median 256 tokens')
log(f'{"rate":>5} {"preemptions":>12} {"recomputed tok":>15} {"TPOT p99":>9} {"ITL p99":>8} {"ITL max":>8} {"E2E p99":>8}')
rows = []
for rate in (2, 3, 4, 5, 6):
    reqs = poisson_workload(rate, 2500, seed=32, output=(256, 0.8, 8, 2048))
    e = Engine(cost, budget=2048, max_seqs=256, kv_capacity=KV)
    run(reqs, [e])
    s = summarize(reqs)
    recomputed = e.prefill_tokens - sum(r.prompt - r.cached for r in reqs)
    rows.append({'rate': rate, 'preemptions': e.preemptions, 'recomputed': recomputed, 'tpot_p99': s['tpot']['p99'],
                 'itl_p99': s['itl']['p99'], 'itl_max': s['itl']['max'], 'e2e_p99': s['e2e']['p99']})
    log(f'{rate:5d} {e.preemptions:12d} {recomputed:15,d} {s["tpot"]["p99"]:7.0f}ms {s["itl"]["p99"]:6.0f}ms {s["itl"]["max"]:6.0f}ms {s["e2e"]["p99"] / 1000:6.1f}s')
OUT['preemption'] = rows

# ---------------------------------------------------------------- 3. head-of-line blocking
log()
log('3. HEAD-OF-LINE BLOCKING: chats at 4 requests/s; one 16,000-token prompt arrives at t = 60 s')
rows = {}
for budget, label in ((2048, 'chunked prefill, 2,048-token budget'), (10**6, 'no chunking (whole prompt in one step)')):
    reqs = poisson_workload(4, 600, seed=33)
    big = Request(10**6, 60_000.0, 16000, 64, origin=10**6)
    e = Engine(cost, budget=budget, max_seqs=256)
    run(sorted(reqs + [big], key=lambda r: r.arrival), [e])
    gaps = [(a, b - a) for r in reqs for a, b in zip(r.token_times, r.token_times[1:])]
    near = [g for a, g in gaps if 59_000 <= a <= 64_000]
    others = [r for r in reqs if 59_000 <= r.arrival <= 64_000]
    rows[label] = {'max_gap_near': max(near), 'p99_gap_all': pct([g for _, g in gaps], 99),
                   'big_ttft': big.first_token - big.arrival,
                   'others_ttft_max': max(r.first_token - r.arrival for r in others)}
    log(f'   {label:40s} longest pause in other streams: {max(near):6.0f} ms; long prompt TTFT {big.first_token - big.arrival:6.0f} ms; '
        f'worst TTFT of chats arriving 59-64 s: {rows[label]["others_ttft_max"]:6.0f} ms')
OUT['hol'] = rows

# ---------------------------------------------------------------- 4. noisy neighbour
log()
log('4. NOISY NEIGHBOUR: tenant A sends chats at 4 requests/s; tenant B sends 40 prompts of 6,000 tokens within 10 s at t = 60 s')
rng = random.Random(34)
rows = {}
for with_b in (False, True):
    reqs = poisson_workload(4, 600, seed=34)
    for r in reqs: r.session = 0
    if with_b:
        for k in range(40):
            reqs.append(Request(50_000 + k, 60_000 + rng.uniform(0, 10_000), 6000, 64, session=1, origin=50_000 + k))
    e = Engine(cost, budget=2048, max_seqs=256)
    run(sorted(reqs, key=lambda r: r.arrival), [e])
    a = [r.first_token - r.arrival for r in reqs if r.session == 0 and 60_000 <= r.arrival <= 90_000]
    rows['with burst' if with_b else 'alone'] = {'p50': pct(a, 50), 'p99': pct(a, 99), 'max': max(a)}
    log(f'   tenant A TTFT for requests arriving 60-90 s, {"B bursting" if with_b else "B silent"}: p50 {pct(a, 50):6.0f} ms, '
        f'p99 {pct(a, 99):6.0f} ms, max {max(a):6.0f} ms')
OUT['noisy'] = rows
log.save(OUT)
