"""A roofline model of one decode step under different multi-GPU layouts (a simulation, not a benchmark).

step time = max(weight + KV bytes read / (p x HBM bandwidth), FLOPs / (p x peak)) + communication
communication (tensor parallel) = 2 all-reduces per layer, two-step model: 2 alpha + 2(p-1)/p x N / B
Everything is peak spec-sheet numbers, so real systems are slower; the comparisons between layouts are the point.
Assumptions (ASSUME below) are stated once and swept where they matter.

1. Tensor parallel scaling: Llama 3.1 70B decode on 1 to 8 H100s (latency of one token, batch 1 and 64).
2. One 8-GPU node for 70B: TP=8 x 1 replica against TP=4 x 2 replicas, latency against throughput.
3. Pipeline parallel across two nodes for 405B: TP=8 x PP=2 against TP=16 over InfiniBand; micro-batches.
4. A pipeline schedule (stage x time grid) for the figure.
Run: python code/multigpu/layout_model.py  -> results/layout_model.json"""
import math
from common import LLAMA, GPUS, LINKS, GB, llama_params, kv_bytes_per_token, save

G = GPUS['H100 SXM']
ASSUME = dict(alpha_nvlink_us=5.0, alpha_ib_us=10.0, usable_fraction=0.90, context_tokens=4096,
              note='alpha = fixed cost per communication step (assumed); context = average tokens in the KV cache per sequence')
NV, IB = LINKS['NVLink 4 (H100, via NVSwitch)'], LINKS['InfiniBand NDR 400 Gb/s (one NIC)']
out = {'assumptions': ASSUME}


def allreduce(N, p, B, alpha):
    return 0.0 if p == 1 else 2 * alpha + 2 * (p - 1) / p * N / B


def step(model, p, batch, ctx=ASSUME['context_tokens'], B=NV, alpha=ASSUME['alpha_nvlink_us'] * 1e-6):
    c = LLAMA[model]
    P = llama_params(c)
    w = 2 * P                                                  # BF16 weights, read once per step
    kv = batch * ctx * kv_bytes_per_token(c)                   # every sequence's cache is read once per step
    t_mem = (w + kv) / (p * G['hbm'])
    t_cmp = 2 * P * batch / (p * G['bf16'])
    t_comm = 2 * c['L'] * allreduce(batch * c['d'] * 2, p, B, alpha)
    return dict(mem_ms=t_mem * 1e3, compute_ms=t_cmp * 1e3, comm_ms=t_comm * 1e3, total_ms=(max(t_mem, t_cmp) + t_comm) * 1e3)


def max_batch(model, p, ctx=ASSUME['context_tokens']):
    c = LLAMA[model]
    free = p * G['mem'] * ASSUME['usable_fraction'] - 2 * llama_params(c)
    return max(0, int(free // (ctx * kv_bytes_per_token(c))))


print('1. TENSOR PARALLEL SCALING, Llama 3.1 70B decode on H100s (NVLink, alpha = 5 us assumed)')
rows = []
for batch in [1, 64]:
    for p in [1, 2, 4, 8]:
        fits = max_batch('Llama 3.1 70B', p) >= batch
        s = step('Llama 3.1 70B', p, batch)
        rows.append(dict(batch=batch, tp=p, fits=fits, **s))
        print(f'   batch {batch:3} TP={p}: ' + (f"memory {s['mem_ms']:6.2f} ms, compute {s['compute_ms']:5.2f} ms, "
              f"all-reduce {s['comm_ms']:5.2f} ms -> {s['total_ms']:6.2f} ms per token"
              if fits else 'does not fit (weights + KV exceed memory)'))
out['tp_scaling_70b'] = rows
for a in [2, 10]:
    s = step('Llama 3.1 70B', 8, 1, alpha=a * 1e-6)
    out[f'tp8_b1_alpha{a}'] = s
    print(f'   sensitivity: TP=8 batch 1 with alpha = {a} us: all-reduce {s["comm_ms"]:.2f} ms, total {s["total_ms"]:.2f} ms')

print('\n2. ONE 8-GPU NODE FOR LLAMA 3.1 70B: one TP=8 replica or two TP=4 replicas (4,096-token contexts)')
curves = {}
for p, reps in [(8, 1), (4, 2), (2, 4)]:
    mb = max_batch('Llama 3.1 70B', p)
    name = f'TP={p} x {reps}'
    if mb == 0:
        print(f'   {name}: no room for any KV cache ({2*llama_params(LLAMA["Llama 3.1 70B"])/GB:.0f} GB of weights on {p} x 80 GB)')
        curves[name] = dict(max_batch=0, points=[])
        continue
    pts = []
    for b in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if b > mb:
            break
        s = step('Llama 3.1 70B', p, b)
        pts.append(dict(batch_per_replica=b, latency_ms=s['total_ms'], node_tokens_per_s=reps * b / s['total_ms'] * 1e3))
    curves[name] = dict(max_batch=mb, points=pts)
    print(f'   {name}: KV room for {mb} sequences per replica ({reps * mb} per node)')
    for q in pts:
        if q['batch_per_replica'] in (1, 16, 64, 128) or q is pts[-1]:
            print(f"      batch {q['batch_per_replica']:3}/replica: {q['latency_ms']:6.2f} ms per token, node {q['node_tokens_per_s']:8,.0f} tokens/s")
out['node_70b'] = curves

print('\n3. LLAMA 3.1 405B (BF16, 812 GB) ON TWO 8-GPU NODES')
c = LLAMA['Llama 3.1 405B']
P = llama_params(c)
res3 = {}
for b in [1, 32]:
    w = 2 * P
    kv = b * ASSUME['context_tokens'] * kv_bytes_per_token(c)
    t_mem16 = (w + kv) / (16 * G['hbm'])
    t_cmp16 = 2 * P * b / (16 * G['bf16'])
    N = b * c['d'] * 2
    # (a) TP=16 across both nodes: every all-reduce crosses InfiniBand (alpha_ib), 2 per layer.
    tp16 = 2 * c['L'] * allreduce(N, 16, IB, ASSUME['alpha_ib_us'] * 1e-6)
    a_lat = max(t_mem16, t_cmp16) + tp16
    # (b) TP=8 inside each node, PP=2 across: half the layers per node, NVLink all-reduces, one IB hop per step.
    stage = max((w + kv) / 2 / (8 * G['hbm']), 2 * P * b / 2 / (8 * G['bf16'])) + c['L'] * allreduce(N, 8, NV, ASSUME['alpha_nvlink_us'] * 1e-6)
    hop = ASSUME['alpha_ib_us'] * 1e-6 + N / IB
    b_lat = 2 * stage + hop
    # with M >= 2 micro-batches in flight, a new micro-batch finishes every stage time
    res3[b] = dict(tp16_ms=a_lat * 1e3, tp16_comm_ms=tp16 * 1e3, pp_stage_ms=stage * 1e3, pp_hop_ms=hop * 1e3,
                   pp_latency_ms=b_lat * 1e3, tp16_tokens_s=b / a_lat, pp_tokens_s_1mb=b / b_lat, pp_tokens_s_2mb=b / stage)
    print(f'   batch {b:2}: TP=16 over InfiniBand: {a_lat*1e3:6.2f} ms per token ({tp16*1e3:.2f} ms of it all-reduce over IB)')
    print(f'             TP=8 x PP=2: stage {stage*1e3:.2f} ms x 2 + IB hop {hop*1e3:.3f} ms = {b_lat*1e3:6.2f} ms per token; '
          f'throughput {b/b_lat:,.0f} tok/s with one micro-batch, {b/stage:,.0f} tok/s with two in flight')
out['pp_405b'] = {str(k): v for k, v in res3.items()}

print('\n4. A PIPELINE SCHEDULE: 4 stages, decode micro-batches (stage time 1 unit, hops ignored)')
K = 4
sched = {}
for M in [1, 4]:
    grid = [['.'] * (3 * K * M + K) for _ in range(K)]
    # each micro-batch generates 3 tokens; a micro-batch's next token can start only after its previous one left stage K
    ready = [0] * M
    busy = [0] * K
    done = 0
    for tok in range(3):
        for m in range(M):
            t = ready[m]
            for s in range(K):
                t = max(t, busy[s])
                grid[s][t] = str(m)
                busy[s] = t + 1
                t += 1
            ready[m] = t
            done = max(done, t)
    util = sum(cell != '.' for row in grid for cell in row[:done]) / (K * done)
    sched[M] = dict(grid=[row[:done] for row in grid], steps=done, utilisation=util, tokens=3 * M)
    print(f'   M={M} micro-batch(es): {3*M} tokens in {done} time units, GPU busy {util:.0%}')
    for s in range(K):
        print(f'      stage {s}: ' + ' '.join(grid[s][:done]))
out['pp_schedule'] = {str(k): v for k, v in sched.items()}
save('layout_model', out)
