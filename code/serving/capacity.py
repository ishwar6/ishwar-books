"""Capacity planning and cost per million tokens on paper, for Llama 3.1 8B on one H100 (Part 6, sections 4 and 5).

Inputs are published specifications (the model's config, NVIDIA's datasheet, vLLM's defaults) plus stated
assumptions. Step times use a roofline bound: each step takes max(FLOPs / peak FLOP/s, bytes / peak bandwidth).
Real engines are slower than this bound, so every capacity here is an upper limit, not a promise.
Then the same bound drives the engine simulator (engine.py) to find the request rate one GPU sustains
inside the SLO, for BF16 and for FP8 weights + FP8 KV cache.
Run: python code/serving/capacity.py  -> results/capacity.json, results/capacity_stdout.txt
"""
import math
from engine import Engine, RooflineCost, poisson_workload, run, summarize
from common import Log, LLAMA8B as M, H100

log = Log('capacity')
OUT = {}
GiB = 2**30

# ---------------------------------------------------------------- 1. memory budget
util = 0.92                                     # vLLM's default --gpu-memory-utilization (config/cache.py)
reserve = 4e9                                   # ASSUMPTION: activations, CUDA graphs, CUDA context
weights_bf16 = M['params'] * 2
kv_tok = M['kv_bytes_per_token_bf16']
kv_space = H100['mem_bytes'] * util - weights_bf16 - reserve
tokens_bf16 = kv_space / kv_tok
log('1. MEMORY on one H100 (80 GB), Llama 3.1 8B in BF16')
log(f'   usable: 80 GB x {util} = {H100["mem_bytes"] * util / 1e9:.1f} GB')
log(f'   weights: {M["params"] / 1e9:.2f} B params x 2 bytes = {weights_bf16 / 1e9:.2f} GB')
log(f'   reserve for activations and runtime (assumption): {reserve / 1e9:.1f} GB')
log(f'   KV cache space: {kv_space / 1e9:.1f} GB / {kv_tok:,} bytes per token = {tokens_bf16:,.0f} tokens')
weights_fp8 = M['params'] * 1
kv_space8 = H100['mem_bytes'] * util - weights_fp8 - reserve
tokens_fp8 = kv_space8 / (kv_tok / 2)
log(f'   with FP8 weights and FP8 KV: weights {weights_fp8 / 1e9:.2f} GB, KV space {kv_space8 / 1e9:.1f} GB '
    f'/ {kv_tok // 2:,} B = {tokens_fp8:,.0f} tokens ({tokens_fp8 / tokens_bf16:.1f}x)')
OUT['memory'] = dict(util=util, reserve=reserve, weights_bf16=weights_bf16, kv_bytes_per_token=kv_tok,
                     kv_space=kv_space, kv_tokens=tokens_bf16, weights_fp8=weights_fp8, kv_space_fp8=kv_space8, kv_tokens_fp8=tokens_fp8)

# ---------------------------------------------------------------- 2. the traffic and Little's law
lam, p_in, p_out = 100.0, 1000, 250              # ASSUMED traffic: requests/s at peak, mean prompt and output tokens
tpot_guess = 0.030
W = 0.5 + p_out * tpot_guess
L = lam * W
kv_need = L * (p_in + p_out / 2) * kv_tok
log()
log(f'2. TRAFFIC (assumed): {lam:.0f} requests/s at peak, {p_in} prompt and {p_out} output tokens on average')
log(f'   prefill work: {lam * p_in:,.0f} tokens/s;  decode work: {lam * p_out:,.0f} tokens/s')
log(f"   Little's law with W = 0.5 s + {p_out} x {tpot_guess * 1000:.0f} ms = {W:.1f} s: L = {lam:.0f} x {W:.1f} = {L:.0f} requests in flight")
log(f'   their KV cache (prompt + half the answer each): {L:.0f} x {p_in + p_out // 2} x {kv_tok // 1024} KiB = {kv_need / 1e9:.1f} GB '
    f'({kv_need / kv_space * 100:.0f}% of the KV space)')
OUT['traffic'] = dict(lam=lam, p_in=p_in, p_out=p_out, W=W, L=L, kv_need=kv_need)

# ---------------------------------------------------------------- 3. per-token costs at the roofline
n_lin = M['params'] - M['vocab'] * M['hidden']           # weights multiplied per token (input embedding is a lookup)
pf_flops = 2 * n_lin
pf_us = pf_flops / H100['bf16_flops'] * 1e6
log()
log('3. ROOFLINE COSTS (H100: 989 TFLOP/s dense BF16, 3.35 TB/s)')
log(f'   prefill: 2 x {n_lin / 1e9:.2f} B = {pf_flops / 1e9:.1f} GFLOP per token -> {pf_us:.1f} us per token at 100% of peak '
    f'({H100["bf16_flops"] / pf_flops:,.0f} tokens/s)')
rows = []
log(f'   decode step, every sequence with 1,125 tokens of context:')
log(f'{"batch":>9} {"bytes read":>11} {"step ms":>8} {"us/token":>9} {"tokens/s":>9}')
for B in (1, 8, 32, 64, 128, 256, 512):
    byts = weights_bf16 + B * 1125 * kv_tok
    flops = B * (pf_flops + 4 * M['layers'] * M['hidden'] * 1125)
    t = max(byts / H100['bw'], flops / H100['bf16_flops'])
    rows.append({'B': B, 'bytes': byts, 'step_ms': t * 1000, 'us_per_token': t / B * 1e6, 'tok_s': B / t})
    log(f'{B:9d} {byts / 1e9:9.1f}GB {t * 1000:8.2f} {t / B * 1e6:9.1f} {B / t:9,.0f}')
OUT['roofline'] = {'prefill_us_per_token': pf_us, 'decode': rows}
gpus_mem = math.ceil(kv_need / kv_space)
gpus_prefill = lam * p_in * pf_us / 1e6
log(f'   lower bound from memory: ceil({kv_need / 1e9:.1f} GB / {kv_space / 1e9:.1f} GB) = {gpus_mem} GPUs')
log(f'   lower bound from prefill arithmetic alone: {lam * p_in:,.0f} tokens/s x {pf_us:.1f} us = {gpus_prefill:.2f} GPU-seconds per second')
OUT['bounds'] = {'gpus_memory': gpus_mem, 'gpu_s_prefill': gpus_prefill}

# ---------------------------------------------------------------- 4. simulated capacity of one GPU
def cost_for(fp8):
    return RooflineCost(n_params=n_lin, weight_bytes=M['params'] * (1 if fp8 else 2),
                        kv_bytes_per_token=kv_tok / (2 if fp8 else 1), attn_flops_per_pair=4 * M['layers'] * M['hidden'],
                        peak_flops=H100['fp8_flops'] if fp8 else H100['bf16_flops'], peak_bw=H100['bw'])

SLO = dict(ttft_slo=500.0, tpot_slo=50.0)
WL = dict(prompt=(800, 0.8, 32, 8192), output=(200, 0.8, 8, 2048))
sample = poisson_workload(1, 20000, seed=11, **WL)
mean_in = sum(r.prompt for r in sample) / len(sample); mean_out = sum(r.output for r in sample) / len(sample)
log()
log(f'4. SIMULATED CAPACITY of one H100 at the roofline (engine.py; budget 8,192 tokens/step, up to 1,024 sequences)')
log(f'   workload: lognormal prompts (mean {mean_in:.0f}) and outputs (mean {mean_out:.0f}); SLO: TTFT <= 500 ms and TPOT <= 50 ms')
log(f'{"format":>10} {"rate":>5} {"tok/s":>7} {"TTFT p99":>9} {"TPOT p99":>9} {"batch":>6} {"preempt":>8} {"SLO met":>8}')
cap = {}
for fp8 in (False, True):
    name = 'FP8' if fp8 else 'BF16'
    kv_cap = int(tokens_fp8 if fp8 else tokens_bf16)
    rows, best = [], None
    for rate in range(10, 125, 5):
        reqs = poisson_workload(rate, max(3000, rate * 120), seed=12, **WL)
        e = Engine(cost_for(fp8), budget=8192, max_seqs=1024, kv_capacity=kv_cap)
        run(reqs, [e])
        s = summarize(reqs, **SLO)
        tr = e.trace[len(e.trace) // 10:]
        s.update(rate=rate, mean_batch=sum(x[2] for x in tr) / len(tr), preemptions=e.preemptions)
        s.update(busy_s=e.busy_ms / 1000, prefill_tokens=e.prefill_tokens, decode_tokens=e.decode_tokens)
        rows.append(s)
        log(f'{name:>10} {rate:5d} {s["output_tok_s"]:7.0f} {s["ttft"]["p99"]:7.0f}ms {s["tpot"]["p99"]:7.1f}ms '
            f'{s["mean_batch"]:6.0f} {e.preemptions:8d} {s["slo_attainment"] * 100:7.1f}%')
        if s['slo_attainment'] >= 0.99: best = rate
        if s['slo_attainment'] < 0.5: break
    cap[name] = {'rows': rows, 'max_rate_99': best}
    log(f'   {name}: highest tested rate with >= 99% of requests inside the SLO: {best} requests/s')
OUT['sim'] = {'workload_mean_in': mean_in, 'workload_mean_out': mean_out, 'slo': SLO, **cap}

# ---------------------------------------------------------------- 5. GPUs for the traffic
log()
log(f'5. GPUs FOR {lam:.0f} requests/s AT PEAK (this workload)')
plan = {}
for name in ('BF16', 'FP8'):
    c = cap[name]['max_rate_99']
    for target in (1.0, 0.7):
        n = math.ceil(lam / (c * target))
        plan[f'{name}_{target}'] = n
        log(f'   {name}: per-GPU bound {c} req/s, planned at {target * 100:.0f}% of it -> ceil({lam:.0f} / {c * target:.1f}) = {n} GPUs'
            + ('' if target == 1 else f', +1 spare for a failure = {n + 1}'))
OUT['plan'] = plan

# ---------------------------------------------------------------- 6. cost per million tokens
log()
log('6. COST PER MILLION TOKENS at the BF16 SLO operating point (GPU price is an ASSUMPTION; scale linearly)')
best = next(r for r in cap['BF16']['rows'] if r['rate'] == cap['BF16']['max_rate_99'])
pf_s = best['prefill_tokens'] * pf_flops / H100['bf16_flops']      # prefill share of busy time, at the roofline
dec_s = best['busy_s'] - pf_s
in_us, out_us = pf_s / best['prefill_tokens'] * 1e6, dec_s / best['decode_tokens'] * 1e6
log(f'   GPU busy {best["busy_s"]:.0f} s for {best["prefill_tokens"]:,} prompt tokens and {best["decode_tokens"]:,} generated tokens')
log(f'   prompt tokens: {pf_s:.0f} s -> {in_us:.1f} us each;  generated tokens: {dec_s:.0f} s -> {out_us:.1f} us each '
    f'(ratio {out_us / in_us:.1f}x)')
cost_rows = []
for price in (2.0, 3.0, 4.0):
    usd_us = price / 3600 / 1e6
    row = {'price': price, 'input_per_M': in_us * usd_us * 1e6, 'output_per_M': out_us * usd_us * 1e6}
    cost_rows.append(row)
    log(f'   ${price:.2f}/GPU-hour: ${row["input_per_M"]:.3f} per 1M prompt tokens, ${row["output_per_M"]:.3f} per 1M generated tokens '
        '(at the roofline, full utilisation)')
per_req = 2.0 / 3600 / best['throughput_rps']
log(f'   per request at $2/GPU-hour: ${per_req:.7f} (${per_req * 1000:.4f} per 1,000 requests)')
for n_out in (250, 4000):
    log(f'   GPU time of one request, 1,000 prompt + {n_out:,} generated tokens: 1,000 x {in_us:.1f} + {n_out:,} x {out_us:.1f} = {1000 * in_us + n_out * out_us:,.0f} us')
for util_ in (0.5, 0.3):
    log(f'   at {util_ * 100:.0f}% average utilisation (traffic is not always at peak): costs x {1 / util_:.1f}')
OUT['cost'] = {'busy_s': best['busy_s'], 'prefill_s': pf_s, 'decode_s': dec_s, 'in_us': in_us, 'out_us': out_us, 'rows': cost_rows}
log.save(OUT)
