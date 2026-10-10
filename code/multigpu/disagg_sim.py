"""A small iteration-level simulator: the same 4 GPUs serving Llama 3.1 8B, colocated or disaggregated.

Each GPU runs a loop of iterations. Cost of one iteration (a roofline with assumed efficiencies, H100 numbers):
    time = max(bytes read / (3.35 TB/s x 0.7), FLOPs / (989 TFLOP/s x 0.5)) + 1 ms fixed overhead
    bytes = 16 GB of weights + the KV cache of every decoding request; FLOPs = 2 x 8B x tokens in the iteration
Layouts (4 GPUs in total, requests spread round-robin):
    colocated, prefill first: a new prompt is prefilled in its own iteration, and running answers wait (vLLM's
                              old default, the interference of Part 1 and Part 3);
    colocated, chunked:       every iteration decodes all running answers plus up to 512 prompt tokens (Part 3);
    disaggregated xP + yD:    prefill GPUs only prefill; the KV cache then moves to the least-loaded decode GPU over
                              NVLink (128 KiB per token at 450 GB/s); decode GPUs only decode.
Workloads: Poisson arrivals. "chat": prompts 1,000 to 3,000 tokens, answers 100 to 400 (uniform).
"long": prompts 8,000 to 16,000 tokens, answers 50 to 200 (documents, RAG). Prefill cost includes causal attention.
A request meets its SLO if its time to first token and its average time per output token are both under
the limits. Three SLOs are reported from the same runs: loose (1 s, 40 ms), tight (0.5 s, 15 ms), strict (1 s, 12 ms).
Goodput (DistServe's definition) = the highest request rate at which at least 90% of requests meet the SLO.
This is a simulation with assumed costs; it shows the mechanism, not any engine's real numbers.
Run: python code/multigpu/disagg_sim.py  -> results/disagg_sim.json"""
import heapq, random
import numpy as np
from common import save

W_BYTES, P = 16e9, 8e9
KV_TOK = 128 * 1024
HBM, PEAK = 3.35e12 * 0.7, 989e12 * 0.5
OVH = 1e-3
NVLINK = 450e9
CHUNK = 512
SLOS = {'loose (TTFT 1 s, TPOT 40 ms)': (1.0, 0.040), 'tight (TTFT 0.5 s, TPOT 15 ms)': (0.5, 0.015),
        'strict (TTFT 1 s, TPOT 12 ms)': (1.0, 0.012)}
DUR = 120.0                                                      # seconds of arrivals per run


L_, D_ = 32, 4096                                                # layers and width, for attention FLOPs


def attn_flops(start, n):
    """Causal attention FLOPs for n new prompt tokens placed after `start` earlier ones (QK^T and AV)."""
    return 4 * L_ * D_ * n * (start + n / 2)


def iter_time(decode_reqs, prefill_tokens, extra_flops=0.0):
    kv = sum(r['ctx'] for r in decode_reqs) * KV_TOK
    toks = len(decode_reqs) + prefill_tokens
    return max((W_BYTES + kv) / HBM, (2 * P * toks + extra_flops) / PEAK) + OVH


WORK = {'chat': (1000, 3000, 100, 400)}


def workload(rate, seed, kind='chat'):
    p0, p1, o0, o1 = WORK[kind]
    rng = random.Random(seed)
    t, reqs = 0.0, []
    while t < DUR:
        t += rng.expovariate(rate)
        reqs.append(dict(id=len(reqs), arrive=t, prompt=rng.randint(p0, p1), out=rng.randint(o0, o1)))
    return reqs


class GPU:
    def __init__(self, mode):
        self.mode, self.wait, self.run, self.t = mode, [], [], 0.0

    def busy(self):
        return bool(self.wait or self.run)

    def step(self, now, emit):
        """Run one iteration starting at `now`; return its end time. emit(req, time) records each output token."""
        self.t = now
        if self.mode in ('prefill_first', 'prefill_only') and self.wait:
            batch, tokens = [], 0
            while self.wait and (not batch or tokens + self.wait[0]['prompt'] <= 8192):
                r = self.wait.pop(0); batch.append(r); tokens += r['prompt']
            end = now + iter_time([], tokens, sum(attn_flops(0, r['prompt']) for r in batch))
            for r in batch:
                r['ctx'] = r['prompt']
                emit(r, end)                                       # prefill produces the first token
                if self.mode == 'prefill_first' and r['left'] > 0:
                    self.run.append(r)
            self.done_prefill = batch
            return end
        if self.mode == 'chunked':
            budget = CHUNK
            parts = []
            while self.wait and budget > 0:
                r = self.wait[0]
                take = min(budget, r['prompt'] - r.get('done', 0))
                parts.append((r, take)); budget -= take
                r['done'] = r.get('done', 0) + take
                if r['done'] >= r['prompt']:
                    self.wait.pop(0)
                else:
                    break
            end = now + iter_time(self.run, sum(t for _, t in parts), sum(attn_flops(r['done'] - t, t) for r, t in parts))
            for r in self.run:
                r['ctx'] += 1; emit(r, end)
            self.run = [r for r in self.run if r['left'] > 0]
            for r, _ in parts:
                if r['done'] >= r['prompt']:
                    r['ctx'] = r['prompt']; emit(r, end)
                    if r['left'] > 0:
                        self.run.append(r)
            return end
        # plain decode iteration
        end = now + iter_time(self.run, 0)
        for r in self.run:
            r['ctx'] += 1; emit(r, end)
        self.run = [r for r in self.run if r['left'] > 0]
        return end


def simulate(layout, rate, seed=0, kind='chat'):
    reqs = workload(rate, seed, kind)
    if layout[0] == 'colocated':
        gpus = [GPU(layout[1]) for _ in range(4)]
        pre, dec = gpus, gpus
    else:
        pre = [GPU('prefill_only') for _ in range(layout[1])]
        dec = [GPU('decode') for _ in range(layout[2])]
        gpus = pre + dec
    for r in reqs:
        r.update(left=r['out'], times=[])

    def emit(r, t):
        r['times'].append(t); r['left'] -= 1

    ev = [(r['arrive'], 0, 'arrive', r['id'], None) for r in reqs]
    heapq.heapify(ev)
    free_at = {id(g): 0.0 for g in gpus}
    running = set()
    rr = 0
    seq = 1

    def kick(g, now):
        nonlocal seq
        if id(g) in running or not g.busy():
            return
        running.add(id(g))
        end = g.step(max(now, free_at[id(g)]), emit)
        free_at[id(g)] = end
        heapq.heappush(ev, (end, seq, 'done', None, g)); seq += 1

    while ev:
        now, _, kind, rid, g = heapq.heappop(ev)
        if kind == 'arrive':
            r = reqs[rid]
            tgt = pre[rr % len(pre)]; rr += 1
            tgt.wait.append(r); kick(tgt, now)
        elif kind == 'done':
            running.discard(id(g))
            if g.mode == 'prefill_only':
                for r in g.done_prefill:
                    if r['left'] > 0:
                        x = r['prompt'] * KV_TOK / NVLINK      # move the KV cache to a decode GPU
                        heapq.heappush(ev, (now + x, seq, 'handoff', r['id'], None)); seq += 1
                g.done_prefill = []
            kick(g, now)
        elif kind == 'handoff':
            r = reqs[rid]
            d = min(dec, key=lambda d: len(d.run))
            d.run.append(r); kick(d, now)
    ttft = np.array([r['times'][0] - r['arrive'] for r in reqs])
    tpot = np.array([(r['times'][-1] - r['times'][0]) / max(1, len(r['times']) - 1) for r in reqs])
    gaps = np.concatenate([np.diff(r['times']) for r in reqs if len(r['times']) > 1])
    att = {k: float(((ttft <= a) & (tpot <= b)).mean()) for k, (a, b) in SLOS.items()}
    return dict(rate=rate, n=len(reqs), ttft_p50=float(np.percentile(ttft, 50)), ttft_p90=float(np.percentile(ttft, 90)),
                tpot_p50=float(np.percentile(tpot, 50)), tpot_p90=float(np.percentile(tpot, 90)),
                gap_p99=float(np.percentile(gaps, 99)), gap_max=float(gaps.max()), slo_attainment=att)


LAYOUTS = {'colocated, prefill first (4 GPUs)': ('colocated', 'prefill_first'),
           'colocated, chunked prefill (4 GPUs)': ('colocated', 'chunked'),
           'disaggregated 1P + 3D': ('disagg', 1, 3),
           'disaggregated 2P + 2D': ('disagg', 2, 2),
           'disaggregated 3P + 1D': ('disagg', 3, 1)}

if __name__ == '__main__':
    WORK['long'] = (8000, 16000, 50, 200)
    out = dict(assumptions=dict(hbm_eff=0.7, flop_eff=0.5, overhead_ms=1, chunk=CHUNK, slos=SLOS, duration_s=DUR,
                                workloads={k: dict(prompt=f'U({a},{b})', output=f'U({c},{d})') for k, (a, b, c, d) in WORK.items()}))
    print('Llama 3.1 8B on 4 H100s (simulated). Goodput = highest request rate with >= 90% of requests inside the SLO.')
    run32 = [dict(ctx=2000)] * 32
    ex = dict(decode_b32_ms=iter_time(run32, 0) * 1e3,
              prefill_2000_ms=iter_time([], 2000, attn_flops(0, 2000)) * 1e3,
              chunk_b32_512_ms=iter_time(run32, 512, attn_flops(1000, 512)) * 1e3)
    out['cost_examples'] = ex
    print(f"cost model: decode step, 32 sequences of 2,000 tokens: {ex['decode_b32_ms']:.1f} ms; prefill of one 2,000-token prompt: "
          f"{ex['prefill_2000_ms']:.1f} ms; decode of 32 + a 512-token chunk (middle of that prompt): {ex['chunk_b32_512_ms']:.1f} ms")
    for kind, rates, show in [('chat', list(range(2, 57, 2)), (8, 16, 24, 32, 40, 48)), ('long', list(range(1, 15)), (2, 4, 6, 8, 10))]:
        a, b_, c, d = WORK[kind]
        print(f'\n=== workload {kind}: prompts {a:,} to {b_:,} tokens, answers {c} to {d} tokens ===')
        out[kind] = {}
        for name, lay in LAYOUTS.items():
            res = [simulate(lay, r, kind=kind) for r in rates]
            good = {k: max([x['rate'] for x in res if x['slo_attainment'][k] >= 0.9], default=0) for k in SLOS}
            out[kind][name] = dict(sweep=res, goodput_rps=good)
            print(f'\n{name}: goodput ' + ', '.join(f'{k.split()[0]} {v} req/s' for k, v in good.items()))
            for x in res:
                if x['rate'] in show:
                    att = list(x['slo_attainment'].values())
                    print(f"   {x['rate']:2} req/s: TTFT p90 {x['ttft_p90']*1e3:7.0f} ms, TPOT p90 {x['tpot_p90']*1e3:5.1f} ms, "
                          f"gap p99 {x['gap_p99']*1e3:4.0f} ms, SLO met loose {att[0]:5.1%} tight {att[1]:5.1%} strict {att[2]:5.1%}")
    save('disagg_sim', out)
