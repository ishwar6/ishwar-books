"""Queueing experiments for Part 6 (simulated, CPU only).

1. M/M/1: a one-at-a-time server with random arrivals and service times, simulated and checked against
   the textbook formulas. Shows why waiting explodes near 100% utilisation.
2. M/D/1: DistServe's equation 1 for prefill-only TTFT, checked against a simulation that uses this
   laptop's measured prefill time.
3. The batching engine (engine.py, step costs fitted to this laptop's measurements) under a load sweep:
   TTFT, TPOT, ITL and end-to-end percentiles, throughput, SLO attainment and goodput.
4. Little's law on the engine: time-averaged requests in the system versus arrival rate x mean latency.
5. Why averages lie: one load point, mean against percentiles.
6. Benchmarking honestly: closed loop versus open loop, coordinated omission with a 3-second stall,
   and bursty arrivals with the same mean rate.
Run: python code/serving/queueing.py  -> results/queueing.json, results/queueing_stdout.txt
"""
import heapq, json, math, random
from engine import Engine, StallEngine, poisson_workload, run, summarize, pct, Request
from common import Log, laptop_cost, RES

log = Log('queueing')
OUT = {}
cost = laptop_cost()
meas = json.loads((RES / 'measure.json').read_text())


# ---------------------------------------------------------------- 1. M/M/1
def mm1(lam, mu, n, seed=0, deterministic=False):
    rng = random.Random(seed)
    t, free_at, resp = 0.0, 0.0, []
    for _ in range(n):
        t += rng.expovariate(lam)
        s = 1 / mu if deterministic else rng.expovariate(mu)
        start = max(t, free_at); free_at = start + s
        resp.append(free_at - t)
    resp = resp[n // 10:]
    return sum(resp) / len(resp), pct(resp, 99)


log('1. M/M/1: one request at a time, service rate mu = 10 per second (mean 100 ms)')
log(f'{"rho":>5} {"lambda/s":>9} {"mean W sim":>11} {"formula":>9} {"p99 sim":>9} {"formula":>9}')
mu, rows = 10.0, []
for rho in (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    lam = rho * mu
    m, p = mm1(lam, mu, 2_000_000, seed=1)
    fm, fp = 1 / (mu - lam), math.log(100) / (mu - lam)
    rows.append({'rho': rho, 'lam': lam, 'mean_ms': m * 1000, 'mean_formula_ms': fm * 1000, 'p99_ms': p * 1000, 'p99_formula_ms': fp * 1000})
    log(f'{rho:5.2f} {lam:9.1f} {m * 1000:9.0f}ms {fm * 1000:7.0f}ms {p * 1000:7.0f}ms {fp * 1000:7.0f}ms')
OUT['mm1'] = rows

# ---------------------------------------------------------------- 2. M/D/1 (DistServe eq. 1)
D = next(r['ms'] for r in meas['prefill'] if r['n'] == 512 and r['c'] == 0) / 1000
log()
log(f'2. M/D/1, DistServe eq. 1: prefill-only server, every prompt 512 tokens, D = {D * 1000:.1f} ms (measured here)')
log(f'{"R/s":>6} {"RD":>5} {"avg TTFT sim":>13} {"eq. 1":>8}')
rows = []
for frac in (0.2, 0.5, 0.8, 0.9, 0.95):
    R = frac / D
    m, _ = mm1(R, 1 / D, 400_000, seed=2, deterministic=True)
    f = D + R * D * D / (2 * (1 - R * D))
    rows.append({'R': R, 'RD': frac, 'sim_ms': m * 1000, 'eq1_ms': f * 1000})
    log(f'{R:6.1f} {frac:5.2f} {m * 1000:11.1f}ms {f * 1000:6.1f}ms')
OUT['md1'] = {'D_ms': D * 1000, 'rows': rows}

# ---------------------------------------------------------------- 3. the batching engine under load
SLO = dict(ttft_slo=1000.0, tpot_slo=50.0)          # chosen for this small model on a laptop
N = 4000
log()
log(f'3. BATCHING ENGINE (step cost fitted to measurements), budget 2048 tokens/step, up to 256 sequences')
log(f'   prompts: median 512 tokens, outputs: median 128 tokens (lognormal); {N} Poisson requests per point')
log(f'   SLO per request: TTFT <= {SLO["ttft_slo"]:.0f} ms and TPOT <= {SLO["tpot_slo"]:.0f} ms')
log(f'{"rate":>5} {"done/s":>7} {"tok/s":>6} {"TTFT p50":>9} {"p99":>7} {"TPOT p50":>9} {"p99":>6} {"ITL p99":>8} {"E2E p50":>8} {"p99":>7} {"SLO ok":>7} {"goodput":>8} {"batch":>6} {"busy":>5}')
rows = []
for rate in (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24):
    reqs = poisson_workload(rate, N, seed=3)
    e = Engine(cost, budget=2048, max_seqs=256)
    run(reqs, [e])
    s = summarize(reqs, **SLO)
    tr = e.trace[len(e.trace) // 10:]
    s['mean_batch'] = sum(x[2] for x in tr) / len(tr)
    s['mean_waiting'] = sum(x[1] for x in tr) / len(tr)
    s['rate'] = rate
    s['busy'] = e.busy_ms / e.now
    rows.append(s)
    log(f'{rate:5d} {s["throughput_rps"]:7.2f} {s["output_tok_s"]:6.0f} {s["ttft"]["p50"]:7.0f}ms {s["ttft"]["p99"]:5.0f}ms '
        f'{s["tpot"]["p50"]:7.1f}ms {s["tpot"]["p99"]:4.0f}ms {s["itl"]["p99"]:6.0f}ms {s["e2e"]["p50"] / 1000:6.1f}s {s["e2e"]["p99"] / 1000:5.1f}s '
        f'{s["slo_attainment"] * 100:6.1f}% {s["goodput_rps"]:8.2f} {s["mean_batch"]:6.1f} {s["busy"] * 100:4.0f}%')
OUT['sweep'] = rows
ok = [r['rate'] for r in rows if r['slo_attainment'] >= 0.9]
OUT['max_rate_90'] = max(ok) if ok else None
log(f'   highest tested rate with >= 90% of requests inside the SLO: {OUT["max_rate_90"]} requests/s')

# ---------------------------------------------------------------- 4. Little's law
log()
log("4. LITTLE'S LAW on the engine: L = lambda x W")
log(f'{"rate":>5} {"lambda seen":>12} {"W (mean E2E)":>13} {"lambda x W":>11} {"L measured":>11}')
rows = []
for rate in (2, 4, 7):
    reqs = poisson_workload(rate, N, seed=3)
    e = Engine(cost, budget=2048, max_seqs=256); run(reqs, [e])
    reqs = reqs[N // 10:]
    W = sum(r.finish - r.arrival for r in reqs) / len(reqs) / 1000
    t0, t1 = reqs[0].arrival, reqs[-1].arrival           # time-average of requests present in [t0, t1]
    ev = sorted([(r.arrival, 1) for r in reqs] + [(r.finish, -1) for r in reqs])
    area, cur, last = 0.0, 0, t0
    for t, d in ev:
        if t > t1: break
        if t >= t0: area += cur * (t - last); last = t
        cur += d
    L = area / (t1 - t0)
    lam_seen = sum(1 for r in reqs if t0 <= r.arrival <= t1) / ((t1 - t0) / 1000)   # the arrival rate this sample really had
    rows.append({'rate': rate, 'lam_seen': lam_seen, 'W_s': W, 'lamW': lam_seen * W, 'L': L})
    log(f'{rate:5d} {lam_seen:10.2f}/s {W:11.2f} s {lam_seen * W:11.2f} {L:11.2f}')
OUT['little'] = rows

# ---------------------------------------------------------------- 5. averages lie
rate = 6
reqs = poisson_workload(rate, N, seed=3)
e = Engine(cost, budget=2048, max_seqs=256); run(reqs, [e])
reqs = reqs[N // 10:]
ttft = sorted(r.first_token - r.arrival for r in reqs)
e2e = sorted(r.finish - r.arrival for r in reqs)
mean = sum(ttft) / len(ttft)
OUT['averages'] = {'rate': rate, 'ttft': ttft, 'mean': mean, 'p50': pct(ttft, 50), 'p90': pct(ttft, 90),
                   'p99': pct(ttft, 99), 'max': ttft[-1], 'share_above_mean': sum(t > mean for t in ttft) / len(ttft),
                   'e2e_mean': sum(e2e) / len(e2e), 'e2e_p50': pct(e2e, 50), 'e2e_p99': pct(e2e, 99)}
a = OUT['averages']
log()
log(f'5. AVERAGES LIE, rate {rate}/s: TTFT mean {a["mean"]:.0f} ms, p50 {a["p50"]:.0f}, p90 {a["p90"]:.0f}, p99 {a["p99"]:.0f}, max {a["max"]:.0f} ms')
log(f'   {(1 - a["share_above_mean"]) * 100:.0f}% of requests are faster than the mean; E2E mean {a["e2e_mean"] / 1000:.2f} s, p50 {a["e2e_p50"] / 1000:.2f} s, p99 {a["e2e_p99"] / 1000:.2f} s')
OUT['averages']['ttft'] = [round(x, 1) for x in ttft]


# ---------------------------------------------------------------- 6. benchmarking: closed loop, open loop, coordinated omission
def closed_loop(engine, users, per_user, period_ms, seed=4, out=(32, 0.3, 8, 64)):
    """Each user wants to send one request every period_ms but never has two in flight (like most load tools).
    Returns (naive latencies from actual send, corrected latencies from the intended send time)."""
    rng = random.Random(seed)
    heap, rid, done_seen = [], 0, 0
    sent = {}
    for u in range(users):
        t = rng.uniform(0, period_ms)
        heapq.heappush(heap, (t, t, u, 0))
    naive, corrected = [], []
    while heap or engine.has_work():
        t_next = heap[0][0] if heap else math.inf
        if engine.has_work() and engine.now < t_next:
            if engine.step() == 0.0: engine.now = t_next
            for r in engine.finished[done_seen:]:
                intended, u, k = sent.pop(r.rid)
                naive.append(r.finish - r.arrival); corrected.append(r.finish - intended)
                if k + 1 < per_user:
                    nxt = intended + period_ms
                    heapq.heappush(heap, (max(nxt, r.finish), nxt, u, k + 1))
            done_seen = len(engine.finished)
            continue
        t, intended, u, k = heapq.heappop(heap)
        engine.now = max(engine.now, t)
        r = Request(rid, t, lognormal_int(rng, 512, 0.8, 32, 4096), lognormal_int(rng, *out))
        sent[rid] = (intended, u, k); rid += 1
        engine.add(r)
    return naive, corrected


from engine import lognormal_int
log()
log('6. BENCHMARKING. Same server, same load (5 requests/s, short answers: median 32 tokens), a 6-second stall at t = 120 s')
users, period = 10, 2000.0                      # 10 users x one request per 2 s = 5 requests/s offered
e = StallEngine(cost, budget=2048, max_seqs=256, at=120_000, stall=6000)
naive, corrected = closed_loop(e, users, 240, period)
reqs = poisson_workload(5, 2400, seed=5, output=(32, 0.3, 8, 64))
e2 = StallEngine(cost, budget=2048, max_seqs=256, at=120_000, stall=6000)
run(reqs, [e2])
openl = [r.finish - r.arrival for r in reqs]
e3 = Engine(cost, budget=2048, max_seqs=256)
reqs3 = poisson_workload(5, 2400, seed=5, output=(32, 0.3, 8, 64)); run(reqs3, [e3])
base = [r.finish - r.arrival for r in reqs3]
rows = {}
for name, xs in (('no stall, open loop', base), ('closed loop, as recorded', naive),
                 ('closed loop, from intended send', corrected), ('open loop (Poisson)', openl)):
    rows[name] = {'n': len(xs), 'p50': pct(xs, 50), 'p99': pct(xs, 99), 'p999': pct(xs, 99.9), 'max': max(xs),
                  'over_3s': sum(x > 3000 for x in xs)}
    log(f'   {name:34s} n={len(xs):5d}  p50 {pct(xs, 50) / 1000:5.2f} s  p99 {pct(xs, 99) / 1000:5.2f} s  '
        f'p99.9 {pct(xs, 99.9) / 1000:5.2f} s  max {max(xs) / 1000:5.2f} s  requests over 3 s: {rows[name]["over_3s"]}')
OUT['coordinated_omission'] = rows

log()
log('   Bursty arrivals: same mean rate, gamma gaps (burstiness < 1 means clumpier traffic)')
rows = []
for rate in (3, 5):
    for b in (1.0, 0.5, 0.25):
        reqs = poisson_workload(rate, N, seed=6, burstiness=b)
        e = Engine(cost, budget=2048, max_seqs=256); run(reqs, [e])
        s = summarize(reqs, **SLO)
        rows.append({'rate': rate, 'burstiness': b, 'ttft_p50': s['ttft']['p50'], 'ttft_p99': s['ttft']['p99'],
                     'tpot_p99': s['tpot']['p99'], 'slo': s['slo_attainment']})
        log(f'   rate {rate:2d}/s burstiness {b:4.2f}: TTFT p50 {s["ttft"]["p50"]:5.0f} ms p99 {s["ttft"]["p99"]:6.0f} ms, '
            f'TPOT p99 {s["tpot"]["p99"]:5.1f} ms, SLO met {s["slo_attainment"] * 100:5.1f}%')
OUT['burstiness'] = rows
log.save(OUT)
