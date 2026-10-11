"""Autoscaling with cold starts (simulated with engine.py, step costs fitted to this laptop's measurements).

Traffic runs at BASE requests/s, then jumps to PEAK at t = 120 s. An autoscaler looks every 10 s.
A new replica serves only after its cold start. Requests go to the ready replica with the fewest outstanding.
Policies:
  static            never scale
  load, cold Xs     target tracking on requests in flight: want ceil(outstanding / TARGET) replicas; new ones
                    are ready X s later (TARGET = 20, about the batch at which queueing.py's engine still met the SLO)
  busy, cold 20s    add a replica when the GPU was busy > 90% of the last interval (the "CPU utilisation" habit)
Run: python code/serving/autoscale.py  -> results/autoscale.json, results/autoscale_stdout.txt
"""
import math
from engine import Engine, poisson_workload, pct
from common import Log, laptop_cost

log = Log('autoscale')
cost = laptop_cost()
BASE, PEAK, T_JUMP, T_END = 6, 24, 120_000, 600_000
CHECK, MAXR, START, TARGET = 10_000, 8, 2, 20


def traffic(seed=41, peak=PEAK):
    a = poisson_workload(BASE, int(BASE * T_JUMP / 1000), seed=seed)
    b = poisson_workload(peak, int(peak * (T_END - T_JUMP) / 1000), seed=seed + 1)
    for r in b: r.arrival += T_JUMP; r.rid += 10**6; r.origin = r.rid
    return a + b


def simulate(policy, cold_ms, peak=PEAK):
    reqs = traffic(peak=peak)
    engines = [Engine(cost, budget=2048, max_seqs=256, rid=k) for k in range(START)]
    ready_at = [0.0] * START
    next_check, last_busy = CHECK, [0.0] * START
    timeline = []
    for r in reqs:
        while next_check <= r.arrival:                  # autoscaler tick
            for e in engines: e.advance_to(next_check)
            ready = [k for k in range(len(engines)) if ready_at[k] <= next_check]
            waiting = sum(len(engines[k].waiting) for k in ready) / max(1, len(ready))
            busy = sum(engines[k].busy_ms - last_busy[k] for k in ready) / (CHECK * max(1, len(ready)))
            last_busy = [e.busy_ms for e in engines]
            pending = len(engines) - len(ready)
            load = sum(engines[k].outstanding() for k in ready)
            if policy == 'load':
                add = min(MAXR, math.ceil(load / TARGET)) - len(engines)
            else:
                add = 1 if policy == 'busy' and busy > 0.9 and pending == 0 else 0
            for _ in range(max(0, min(add, MAXR - len(engines)))):
                engines.append(Engine(cost, budget=2048, max_seqs=256, rid=len(engines)))
                engines[-1].now = next_check + cold_ms
                ready_at.append(next_check + cold_ms); last_busy.append(0.0)
            timeline.append({'t': next_check / 1000, 'ready': len(ready), 'total': len(engines), 'waiting': waiting, 'busy': busy, 'load': load})
            next_check += CHECK
        for e in engines: e.advance_to(r.arrival)
        ready = [k for k in range(len(engines)) if ready_at[k] <= r.arrival]
        k = min(ready, key=lambda k: engines[k].outstanding())
        engines[k].add(r)
    for e in engines:
        while e.has_work():
            if e.step() == 0.0: break
    win = []
    for w0 in range(0, T_END, 20_000):
        xs = [r.first_token - r.arrival for r in reqs if w0 <= r.arrival < w0 + 20_000]
        win.append({'t': w0 / 1000, 'p50': pct(xs, 50), 'p99': pct(xs, 99)})
    after = [r.first_token - r.arrival for r in reqs if r.arrival >= T_JUMP]
    gpu_s = sum((T_END - min(T_END, ra)) / 1000 for ra in ready_at)
    return {'windows': win, 'timeline': timeline, 'replicas_end': len(engines), 'ttft_p50_after': pct(after, 50),
            'ttft_p99_after': pct(after, 99), 'replica_seconds': gpu_s,
            'scale_times': [round(ra / 1000) for ra in ready_at[START:]]}


log(f'Traffic: {BASE} requests/s, then {PEAK} requests/s from t = {T_JUMP // 1000} s to {T_END // 1000} s. Start with {START} replicas, at most {MAXR}.')
OUT = {}
for name, pol, cold in (('static', 'static', 0), ('load, cold 180s', 'load', 180_000), ('load, cold 60s', 'load', 60_000),
                        ('load, cold 20s', 'load', 20_000), ('busy, cold 20s', 'busy', 20_000)):
    r = simulate(pol, cold)
    OUT[name] = r
    log(f'{name:>17}: replicas at end {r["replicas_end"]}, ready at t = {r["scale_times"]} s; after the jump TTFT p50 '
        f'{r["ttft_p50_after"] / 1000:6.2f} s, p99 {r["ttft_p99_after"] / 1000:6.1f} s; replica-seconds {r["replica_seconds"]:5.0f}')
log()
log(f'Steady traffic, {BASE} requests/s all the time (no jump):')
for name, pol in (('load, steady', 'load'), ('busy, steady', 'busy')):
    r = simulate(pol, 20_000, peak=BASE)
    OUT[name] = r
    log(f'{name:>17}: replicas at end {r["replicas_end"]}, ready at t = {r["scale_times"]} s; TTFT p99 {r["ttft_p99_after"] / 1000:6.1f} s; '
        f'replica-seconds {r["replica_seconds"]:5.0f}')
log.save(OUT)
