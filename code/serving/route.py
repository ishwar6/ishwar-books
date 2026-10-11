"""Routing across replicas (simulated): which replica should get the next request?

Four replicas of the engine in engine.py (step costs fitted to this laptop's measurements), each with its own
prefix cache. Requests share long prefixes (a system prompt, a document, a chat history): a request that lands
on a replica already holding its prefix prefills only its new tokens.

Policies:
  random, round_robin            ignore both load and cache
  least_outstanding              fewest queued + running requests (join the shortest queue)
  power_of_two                   sample two replicas, take the less loaded one
  prefix_hash                    always send a prefix to the same replica (session stickiness / consistent hashing)
  cache_aware                    the rule of SGLang's router (sgl-model-gateway, policies/cache_aware.rs):
                                 if the load is imbalanced, shortest queue; otherwise the replica whose
                                 (approximate) tree matches the most, or the emptiest tree if the match is small
Run: python code/serving/route.py  -> results/route.json, results/route_stdout.txt
"""
import random, hashlib
from collections import OrderedDict
from engine import Engine, Request, run, summarize, lognormal_int
from common import Log, laptop_cost

log = Log('route')
cost = laptop_cost()
R = 4
PREFIX_LEN = 2048
CACHE = 16 * PREFIX_LEN            # each replica can keep 16 prefixes


def workload(rate, n, seed, n_prefix=200, zipf=1.1, hot=None):
    """n requests; prefix popularity is Zipf(zipf) over n_prefix prefixes, or one hot prefix takes `hot` of traffic."""
    rng = random.Random(seed)
    w = [1 / (k + 1) ** zipf for k in range(n_prefix)]
    t, out = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate / 1000)
        p = 0 if hot and rng.random() < hot else rng.choices(range(n_prefix), w)[0]
        suffix = lognormal_int(rng, 128, 0.6, 16, 1024)
        out.append(Request(i, t, PREFIX_LEN + suffix, lognormal_int(rng, 64, 0.6, 8, 512), prefix=p, prefix_len=PREFIX_LEN, origin=i))
    return out


class Router:
    def __init__(self, policy, seed=0, cache_threshold=0.3, abs_thr=64, rel_thr=1.5):
        self.policy, self.rng, self.rr = policy, random.Random(seed), 0
        self.trees = [OrderedDict() for _ in range(R)]       # the router's guess of each replica's cache
        self.ct, self.abs_thr, self.rel_thr = cache_threshold, abs_thr, rel_thr
        self.decisions = {'balance': 0, 'match': 0, 'emptiest': 0}

    def remember(self, i, r):
        t = self.trees[i]
        t[r.prefix] = True; t.move_to_end(r.prefix)
        while len(t) > CACHE // PREFIX_LEN:
            t.popitem(last=False)

    def __call__(self, r, engines):
        load = [e.outstanding() for e in engines]
        p = self.policy
        if p == 'random':
            i = self.rng.randrange(R)
        elif p == 'round_robin':
            i = self.rr % R; self.rr += 1
        elif p == 'least_outstanding':
            m = min(load); i = self.rng.choice([k for k in range(R) if load[k] == m])
        elif p == 'power_of_two':
            a, b = self.rng.sample(range(R), 2); i = a if load[a] <= load[b] else b
        elif p == 'prefix_hash':
            i = int(hashlib.sha256(str(r.prefix).encode()).hexdigest(), 16) % R
        elif p.startswith('cache_aware'):
            if max(load) - min(load) > self.abs_thr and max(load) > self.rel_thr * min(load):
                m = min(load); i = self.rng.choice([k for k in range(R) if load[k] == m]); self.decisions['balance'] += 1
            else:
                match = [r.prefix_len / r.prompt if r.prefix in self.trees[k] else 0.0 for k in range(R)]
                if max(match) > self.ct:
                    i = match.index(max(match)); self.decisions['match'] += 1
                else:
                    sizes = [len(t) for t in self.trees]; i = sizes.index(min(sizes)); self.decisions['emptiest'] += 1
        self.remember(i, r)
        return i


POLICIES = ['random', 'round_robin', 'least_outstanding', 'power_of_two', 'prefix_hash', 'cache_aware', 'cache_aware_abs8']
OUT = {}
RATES = {'zipf': 20, 'hot': 20}            # total requests/s, chosen so four replicas are busy but not overloaded
for scen, kw in (('zipf', {}), ('hot', {'hot': 0.5})):
    OUT[scen] = {}
    rate = RATES[scen]
    log()
    log(f'SCENARIO {scen}: {R} replicas, {rate} requests/s in total, prefixes of {PREFIX_LEN} tokens, '
        + ('200 prefixes with Zipf(1.1) popularity' if scen == 'zipf' else 'one prefix takes 50% of traffic, the rest Zipf over 200')
        + f'; each replica caches {CACHE // PREFIX_LEN} prefixes')
    log(f'{"policy":>18} {"hit rate":>9} {"TTFT p50":>9} {"p99":>8} {"E2E p99":>8} {"busiest/mean":>13} {"preempt":>8}')
    for pol in POLICIES:
        reqs = workload(rate, 6000, seed=21, **kw)
        engines = [Engine(cost, budget=2048, max_seqs=256, cache_capacity=CACHE, rid=k) for k in range(R)]
        router = Router(pol.replace('_abs8', ''), seed=3, abs_thr=8 if pol.endswith('abs8') else 64)
        run(reqs, engines, route=router)
        s = summarize(reqs)
        kept = sorted(reqs, key=lambda r: r.arrival)[len(reqs) // 10:]
        hit = sum(r.cached for r in kept) / sum(r.prompt for r in kept)
        per = [sum(1 for r in kept if r.replica == k) for k in range(R)]
        imb = max(per) / (sum(per) / R)
        OUT[scen][pol] = {'hit_rate': hit, 'ttft_p50': s['ttft']['p50'], 'ttft_p99': s['ttft']['p99'], 'e2e_p99': s['e2e']['p99'],
                          'imbalance': imb, 'per_replica': per, 'decisions': dict(router.decisions),
                          'preemptions': sum(e.preemptions for e in engines)}
        log(f'{pol:>18} {hit * 100:8.1f}% {s["ttft"]["p50"]:7.0f}ms {s["ttft"]["p99"]:6.0f}ms {s["e2e"]["p99"] / 1000:6.1f}s '
            f'{imb:13.2f} {OUT[scen][pol]["preemptions"]:8d}')
log()
log('cache_aware uses the gateway defaults (cache_threshold 0.3, balance_abs_threshold 64, balance_rel_threshold 1.5);')
log('cache_aware_abs8 is the same rule with balance_abs_threshold 8, so it falls back to shortest-queue sooner.')
log.save(OUT)
