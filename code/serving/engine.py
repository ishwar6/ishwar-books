"""A small discrete-event model of an LLM serving engine, for the queueing, benchmarking, routing and
failure experiments of Part 6. It is a model, not vLLM or SGLang, but it follows their main loop:

  every step: each running request that is past prefill decodes one token; the rest of a token budget
  goes to prefill (chunked, oldest first); new requests join while there is room (continuous batching);
  if the KV memory runs out, the newest running request is preempted and later recomputed.

Each step costs t = a + b * new_tokens + g * prefill_attention_pairs + k * kv_tokens_read milliseconds,
a model fitted to real measurements (measure.py), or a roofline bound (RooflineCost). Times are milliseconds throughout.
"""
import heapq, math, random
from dataclasses import dataclass, field
from collections import OrderedDict


# ---------------------------------------------------------------- step cost models
@dataclass
class LinearCost:
    a: float          # fixed ms per step (launches, weight reads at small batch)
    b: float          # ms per new token computed in the step
    g: float          # ms per prefill attention pair (new prompt token x earlier token)
    k: float = 0.0    # ms per cached token read (decode attention reads every sequence's whole cache)
    def __call__(self, new, pairs, kv_read=0, dec_pairs=0):
        return self.a + self.b * new + self.g * pairs + self.k * kv_read


@dataclass
class RooflineCost:
    """Best case on a GPU: a step takes max(compute time, memory time). Ignores every overhead."""
    n_params: float          # non-embedding weights
    weight_bytes: float      # bytes read per step (all weights)
    kv_bytes_per_token: float
    attn_flops_per_pair: float
    peak_flops: float
    peak_bw: float
    overhead_ms: float = 0.0
    def __call__(self, new, pairs, kv_read=0, dec_pairs=0):
        flops = 2 * self.n_params * new + self.attn_flops_per_pair * (pairs + dec_pairs)
        byts = self.weight_bytes + self.kv_bytes_per_token * kv_read
        return self.overhead_ms + 1000 * max(flops / self.peak_flops, byts / self.peak_bw)


# ---------------------------------------------------------------- requests and workloads
@dataclass
class Request:
    rid: int
    arrival: float
    prompt: int
    output: int
    prefix: int = -1          # id of a shared prefix (system prompt, document), -1 = none
    prefix_len: int = 0
    session: int = -1
    deadline: float = math.inf
    attempt: int = 0
    origin: int = -1          # id of the first attempt (for retries)
    # filled in by the engine
    need: int = 0             # prompt tokens still to prefill
    done_prefill: int = 0
    generated: int = 0
    first_token: float = None
    finish: float = None
    token_times: list = field(default_factory=list)
    cached: int = 0           # prompt tokens served from the prefix cache
    preempted: int = 0
    replica: int = -1
    cancelled: bool = False
    def kv(self):
        return self.done_prefill + self.generated


def lognormal_int(rng, median, sigma, lo, hi):
    return int(min(hi, max(lo, round(median * math.exp(rng.gauss(0, sigma))))))


def poisson_workload(rate_per_s, n, seed=0, prompt=(512, 0.8, 32, 4096), output=(128, 0.8, 8, 1024),
                     burstiness=1.0):
    """n requests with exponential gaps (Poisson arrivals). burstiness < 1 gives gamma gaps with the same mean
    but more clumping (the same knob as vllm bench serve --burstiness)."""
    rng = random.Random(seed)
    t, reqs = 0.0, []
    mean_gap = 1000.0 / rate_per_s
    for i in range(n):
        gap = rng.gammavariate(burstiness, mean_gap / burstiness)
        t += gap
        reqs.append(Request(i, t, lognormal_int(rng, *prompt), lognormal_int(rng, *output), origin=i))
    return reqs


# ---------------------------------------------------------------- one engine replica
class Engine:
    def __init__(self, cost, budget=2048, max_seqs=256, kv_capacity=10**9, cache_capacity=0, rid=0,
                 cancel_on_timeout=False):
        self.cost, self.budget, self.max_seqs, self.kv_capacity = cost, budget, max_seqs, kv_capacity
        self.cache = OrderedDict()           # prefix id -> length, in LRU order (last = most recent)
        self.cache_capacity, self.cache_used = cache_capacity, 0
        self.waiting, self.running = [], []
        self.now, self.id = 0.0, rid
        self.cancel_on_timeout = cancel_on_timeout
        self.finished, self.preemptions, self.steps, self.busy_ms = [], 0, 0, 0.0
        self.prefill_tokens = self.decode_tokens = 0
        self.trace = []                      # (time, waiting, running, kv used) after every step

    # load signals a router or autoscaler can read
    def outstanding(self):
        return len(self.waiting) + len(self.running)
    def kv_used(self):
        return sum(r.kv() for r in self.running)

    def add(self, r):
        r.need, r.replica = r.prompt, self.id
        self.waiting.append(r)

    def _cache_lookup(self, r):
        if r.prefix >= 0 and r.prefix in self.cache:
            self.cache.move_to_end(r.prefix)
            return self.cache[r.prefix]
        return 0

    def _cache_insert(self, r):
        if r.prefix < 0 or self.cache_capacity <= 0 or r.prefix_len > self.cache_capacity:
            return
        if r.prefix in self.cache:
            self.cache.move_to_end(r.prefix); return
        while self.cache_used + r.prefix_len > self.cache_capacity:
            _, n = self.cache.popitem(last=False); self.cache_used -= n
        self.cache[r.prefix] = r.prefix_len; self.cache_used += r.prefix_len

    def has_work(self):
        return bool(self.waiting or self.running)

    def step(self):
        """Run one engine step starting at self.now; returns the step length in ms."""
        if self.cancel_on_timeout:          # a client that gave up: free its memory and drop it
            for q in (self.waiting, self.running):
                for r in [r for r in q if r.deadline < self.now]:
                    r.cancelled = True; q.remove(r)
        decoding = [r for r in self.running if r.done_prefill >= r.need]
        # memory for one more token per decoding request; preempt newest first (vLLM's recompute policy)
        while decoding and self.kv_used() + len(decoding) > self.kv_capacity:
            victim = max(self.running, key=lambda r: r.arrival)
            self.running.remove(victim)
            if victim in decoding:
                decoding.remove(victim)
            victim.need = victim.prompt + victim.generated   # recompute prompt + tokens so far
            victim.done_prefill, victim.cached = 0, 0
            victim.preempted += 1; self.preemptions += 1
            self.waiting.insert(0, victim)
        new, pairs = len(decoding), 0
        dec_pairs = kv_read = sum(r.kv() + 1 for r in decoding)   # decode: each sequence reads its whole cache once
        left = self.budget - new
        chunks = []
        for r in [r for r in self.running if r.done_prefill < r.need]:      # continue partial prefills
            if left <= 0: break
            n = min(left, r.need - r.done_prefill)
            chunks.append((r, n)); left -= n
        free = self.kv_capacity - self.kv_used() - len(decoding) - sum(n for _, n in chunks)
        while self.waiting and left > 0 and len(self.running) < self.max_seqs:
            r = self.waiting[0]
            hit = min(self._cache_lookup(r), r.need - 1)       # prompt tokens already in the prefix cache
            n = min(left, r.need - hit)
            if n > free:                     # not enough KV memory for its first chunk: wait
                break
            r.cached = r.done_prefill = hit
            self.waiting.pop(0); self.running.append(r)
            chunks.append((r, n)); left -= n; free -= n
        for r, n in chunks:
            c = r.done_prefill
            new += n; pairs += n * c + n * (n + 1) // 2; kv_read += c + n
        if new == 0:
            return 0.0
        dt = self.cost(new, pairs, kv_read, dec_pairs)
        t_end = self.now + dt
        self.decode_tokens += len(decoding); self.prefill_tokens += sum(n for _, n in chunks)
        for r in decoding:
            r.generated += 1; r.token_times.append(t_end)
        for r, n in chunks:
            r.done_prefill += n
            if r.done_prefill >= r.need:     # prefill finished: this step also yields a token
                self._cache_insert(r)
                if r.first_token is None:
                    r.first_token = t_end
                r.generated += 1; r.token_times.append(t_end)
        for r in [r for r in self.running if r.generated >= r.output]:
            r.finish = t_end; self.running.remove(r); self.finished.append(r)
        self.now = t_end; self.steps += 1; self.busy_ms += dt
        self.trace.append((t_end, len(self.waiting), len(self.running), self.kv_used()))
        return dt

    def advance_to(self, t):
        """Run steps until the engine's clock reaches t (or it runs out of work)."""
        while self.now < t:
            if not self.has_work():
                self.now = t; return
            if self.step() == 0.0:            # waiting but blocked (memory): idle until t
                self.now = t; return


class StallEngine(Engine):
    """An engine that freezes once for `stall` ms at time `at` (a GPU hiccup, a long pause, a host problem)."""
    def __init__(self, *a, at=None, stall=0.0, **k):
        super().__init__(*a, **k); self.at, self.stall, self.done_stall = at, stall, False
    def step(self):
        if self.at is not None and not self.done_stall and self.now >= self.at:
            self.done_stall = True; self.now += self.stall; self.busy_ms += self.stall
        return super().step()


# ---------------------------------------------------------------- a cluster of replicas behind a router
def run(requests, engines, route=None, retry=None, horizon=math.inf):
    """Feed requests (sorted by arrival) to the engines. route(r, engines) -> engine index.
    retry(r) -> a new Request to submit later, or None. Returns every attempt that was submitted."""
    route = route or (lambda r, es: 0)
    heap = [(r.arrival, r.rid, r) for r in requests]
    heapq.heapify(heap)
    submitted, next_id = [], max((r.rid for r in requests), default=0) + 1
    timeouts = []                                      # (deadline, attempt) for retry generation
    while heap or timeouts:
        if timeouts and (not heap or timeouts[0][0] <= heap[0][0]):
            t, _, r = heapq.heappop(timeouts)
            for e in engines: e.advance_to(t)
            if r.finish is None or r.finish > r.deadline:
                nr = retry(r) if retry else None
                if nr is not None:
                    nr.rid = next_id; next_id += 1
                    heapq.heappush(heap, (nr.arrival, nr.rid, nr))
            continue
        t, _, r = heapq.heappop(heap)
        if t > horizon: break
        for e in engines: e.advance_to(t)
        e = engines[route(r, engines)]
        e.add(r); submitted.append(r)
        if retry and r.deadline < math.inf:
            heapq.heappush(timeouts, (r.deadline, r.rid, r))
    for e in engines:
        while e.has_work():
            if e.step() == 0.0: break
    return submitted


# ---------------------------------------------------------------- metrics
def pct(xs, p):
    if not xs: return float('nan')
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(reqs, warmup_frac=0.1, ttft_slo=None, tpot_slo=None):
    """Latency statistics over finished requests, skipping the first warmup_frac of arrivals."""
    reqs = sorted(reqs, key=lambda r: r.arrival)
    reqs = reqs[int(len(reqs) * warmup_frac):]
    done = [r for r in reqs if r.finish is not None]
    ttft = [r.first_token - r.arrival for r in done]
    tpot = [(r.finish - r.first_token) / (r.output - 1) for r in done if r.output > 1]
    itl = [b - a for r in done for a, b in zip(r.token_times, r.token_times[1:])]
    e2e = [r.finish - r.arrival for r in done]
    span = (max(r.finish for r in done) - min(r.arrival for r in reqs)) / 1000 if done else float('nan')
    out = {'n': len(reqs), 'finished': len(done),
           'throughput_rps': len(done) / span, 'output_tok_s': sum(r.output for r in done) / span}
    for name, xs in (('ttft', ttft), ('tpot', tpot), ('itl', itl), ('e2e', e2e)):
        out[name] = {'mean': sum(xs) / len(xs) if xs else float('nan'),
                     'p50': pct(xs, 50), 'p90': pct(xs, 90), 'p99': pct(xs, 99), 'max': max(xs) if xs else float('nan')}
    if ttft_slo is not None:
        ok = [r for r in done if r.first_token - r.arrival <= ttft_slo and
              (r.output <= 1 or (r.finish - r.first_token) / (r.output - 1) <= tpot_slo)]
        out['slo_attainment'] = len(ok) / len(reqs)
        out['goodput_rps'] = len(ok) / span
    return out
