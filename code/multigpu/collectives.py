"""Collectives, step by step, in plain NumPy (a simulation: the "GPUs" are list entries).

1. Ring all-reduce on 4 ranks: reduce-scatter (p-1 steps) then all-gather (p-1 steps), printing every step,
   checking the result equals the plain sum, and counting the bytes each rank sends.
2. All-gather, reduce-scatter and all-to-all on the same 4 ranks.
3. The cost formula T = 2(p-1) * (alpha + N / (p * B)) for a 70B model's per-token all-reduces,
   for several link types and assumed per-step latencies alpha.
Run: python code/multigpu/collectives.py  -> results/collectives.json"""
import numpy as np
from common import LLAMA, LINKS, save

out = {}


def ring_allreduce(data, verbose=False):
    """data: list of p equal-length arrays (one per rank). Returns (result per rank, bytes sent per rank, trace)."""
    p = len(data)
    buf = [np.array(d, dtype=float).reshape(p, -1).copy() for d in data]   # each rank's vector cut into p chunks
    sent = [0] * p
    trace = []
    # Reduce-scatter: at step s, rank r sends chunk (r - s) % p to rank r+1, which adds it to its own copy.
    for s in range(p - 1):
        msgs = [(r, (r - s) % p, buf[r][(r - s) % p].copy()) for r in range(p)]
        for r, c, chunk in msgs:
            buf[(r + 1) % p][c] += chunk
            sent[r] += chunk.nbytes
        trace.append(('reduce-scatter', s + 1, [b.copy() for b in buf]))
    # Now rank r owns the complete sum of chunk (r + 1) % p. All-gather: pass the finished chunks around.
    for s in range(p - 1):
        msgs = [(r, (r + 1 - s) % p, buf[r][(r + 1 - s) % p].copy()) for r in range(p)]
        for r, c, chunk in msgs:
            buf[(r + 1) % p][c] = chunk
            sent[r] += chunk.nbytes
        trace.append(('all-gather', s + 1, [b.copy() for b in buf]))
    return [b.ravel() for b in buf], sent, trace


print('1. RING ALL-REDUCE, 4 ranks, each holding 4 numbers (one chunk per rank)')
p = 4
data = [np.array([1, 2, 3, 4]) * (r + 1) for r in range(p)]          # rank r holds (r+1) * [1,2,3,4]
for r, d in enumerate(data):
    print(f'   start  rank {r}: {d.tolist()}')
res, sent, trace = ring_allreduce(data)
steps = []
for phase, s, bufs in trace:
    rows = [b.ravel().astype(int).tolist() for b in bufs]
    steps.append(dict(phase=phase, step=s, ranks=rows))
    print(f'   {phase:14} step {s}: ' + '  '.join(f'r{r}={row}' for r, row in enumerate(rows)))
target = np.sum(data, axis=0)
print(f'   every rank ends with {res[0].astype(int).tolist()} = plain sum {target.tolist()}: {all((x == target).all() for x in res)}')
out['ring_trace'] = dict(start=[d.tolist() for d in data], steps=steps)

# Bytes: check the 2(p-1)/p formula with a bigger vector.
for p in [2, 4, 8]:
    N = 1 << 20
    vecs = [np.random.default_rng(r).standard_normal(N // 8) for r in range(p)]   # N bytes of float64
    res, sent, _ = ring_allreduce(vecs)
    ok = np.allclose(res[0], np.sum(vecs, axis=0))
    print(f'   p={p}: each rank sent {sent[0]/N:.3f} x N bytes (formula 2(p-1)/p = {2*(p-1)/p:.3f}); correct: {ok}')
    out[f'ring_bytes_p{p}'] = dict(sent_over_N=sent[0] / N, formula=2 * (p - 1) / p, correct=bool(ok))

print('\n2. THE OTHER COLLECTIVES on 4 ranks')
p = 4
x = [np.array([10 * r + c for c in range(p)]) for r in range(p)]       # rank r holds [10r, 10r+1, 10r+2, 10r+3]
ag = [np.concatenate([x[r][r:r + 1] for r in range(p)])] * p           # all-gather of each rank's one piece
rs = [np.sum([x[q][r] for q in range(p)]) for r in range(p)]           # reduce-scatter: rank r gets sum of piece r
a2a = [np.array([x[q][r] for q in range(p)]) for r in range(p)]        # all-to-all: rank r gets piece r of everyone
for r in range(p):
    print(f'   rank {r}: has {x[r].tolist()}   all-gather(own piece {x[r][r]}) -> {ag[r].tolist()}   '
          f'reduce-scatter -> {int(rs[r])}   all-to-all -> {a2a[r].tolist()}')
out['collectives_demo'] = dict(start=[v.tolist() for v in x], all_gather=ag[0].tolist(),
                               reduce_scatter=[int(v) for v in rs], all_to_all=[v.tolist() for v in a2a])

print('\n3. COST OF THE ALL-REDUCES IN ONE DECODE STEP, Llama 3.1 70B (80 layers, d = 8192, BF16)')
c = LLAMA['Llama 3.1 70B']
calls = 2 * c['L']                                                     # two all-reduces per layer (Megatron)
print(f'   {calls} all-reduces per step; each carries N = batch x d x 2 bytes')
print('   ring:      2(p-1) steps,  T = 2(p-1) * (alpha + N/(p B))')
print('   two-step:  2 steps (reduce-scatter + all-gather done in one hop each, as in one-shot/MultiShot designs),'
      '  T = 2 alpha + 2(p-1)/p * N/B')
ring = lambda N, p, B, a: 2 * (p - 1) * (a + N / (p * B))
two = lambda N, p, B, a: 2 * a + 2 * (p - 1) / p * N / B
table = {}
for batch in [1, 64]:
    N = batch * c['d'] * 2
    print(f'   batch {batch}: N = {N/1024:.0f} KiB per all-reduce; milliseconds per decode step for all {calls}:')
    for link, B in LINKS.items():
        for p in [2, 4, 8]:
            for a in [2e-6, 5e-6, 10e-6]:
                table[f'{link}|p{p}|a{a*1e6:.0f}|B{batch}'] = dict(
                    ring_ms=calls * ring(N, p, B, a) * 1e3, two_step_ms=calls * two(N, p, B, a) * 1e3,
                    bytes_only_ms=calls * 2 * (p - 1) / p * N / B * 1e3)
        cells = []
        for p in [2, 8]:
            r = [table[f'{link}|p{p}|a{a}|B{batch}'] for a in (2, 10)]
            cells.append(f"TP={p}: ring {r[0]['ring_ms']:5.2f}-{r[1]['ring_ms']:5.2f}, two-step {r[0]['two_step_ms']:5.2f}-"
                         f"{r[1]['two_step_ms']:5.2f}, bytes {r[0]['bytes_only_ms']:5.3f}")
        print(f'     {link:34} ' + ';  '.join(cells))
print('   (ranges: alpha = 2 to 10 microseconds per step, an assumption; B = peak per-direction bandwidth)')
out['cost_70b'] = table
out['cost_note'] = 'alpha (fixed cost per communication step) is an assumption swept over 2, 5 and 10 microseconds; B is peak per-direction link bandwidth'

print('\n4. EXPERT PARALLEL ALL-TO-ALL BYTES, DeepSeek-V3 shapes (hidden 7,168, top-8, FP8 dispatch, BF16 combine)')
from common import DSV3
d, k = DSV3['d'], DSV3['topk']
moe_layers = DSV3['L'] - DSV3['dense_layers']
per_tok_dispatch, per_tok_combine = k * d * 1, k * d * 2
print(f'   per token per MoE layer: dispatch {k} x {d:,} x 1 B = {per_tok_dispatch:,} B, combine {k} x {d:,} x 2 B = {per_tok_combine:,} B')
print(f'   per token, all {moe_layers} MoE layers: {(per_tok_dispatch + per_tok_combine) * moe_layers / 1e6:.1f} MB')
# DeepEP v1.2.1 README, low-latency kernels: 128 tokens per batch, EP8: dispatch 77 us at 98 GB/s, combine 114 us at 127 GB/s.
T = 128
disp, comb = T * per_tok_dispatch, T * per_tok_combine
t_d, t_c = disp / 98e9, comb / 127e9
print(f'   batch of {T} tokens: dispatch {disp/1e6:.2f} MB / 98 GB/s = {t_d*1e6:.0f} us (DeepEP reports 77 us); '
      f'combine {comb/1e6:.2f} MB / 127 GB/s = {t_c*1e6:.0f} us (reports 114 us)')
print(f'   one decode step, {moe_layers} MoE layers at the reported EP8 latencies: {moe_layers*(77+114)/1e3:.1f} ms of all-to-all '
      f'(at EP256: {moe_layers*(194+360)/1e3:.1f} ms), unless hidden behind compute')
out['moe_a2a'] = dict(per_token_dispatch_B=per_tok_dispatch, per_token_combine_B=per_tok_combine, moe_layers=moe_layers,
                      batch128_dispatch_us=t_d * 1e6, batch128_combine_us=t_c * 1e6,
                      step_ms_ep8=moe_layers * (77 + 114) / 1e3, step_ms_ep256=moe_layers * (194 + 360) / 1e3)
save('collectives', out)
