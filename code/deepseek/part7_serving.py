"""Part 7: speed ceilings and DeepSeek's own serving numbers, recomputed from their sources.

1. Decode speed ceiling = memory bandwidth / bytes read per token (MoE versus dense).
2. The ISCA 2025 paper's communication limit for expert parallelism (14.76 ms per token).
3. Expert placement in DeepSeek's prefill and decode units (V3 paper, Dec 2024, and the Feb 2025 overview).
4. The Feb 2025 "Day 6" statistics: cost, theoretical revenue and the 545% margin.
Inputs are quoted in research/04-hardware-and-serving.md with their sources; this script only does the arithmetic."""
from common import Log, save

log = Log('part7_serving')
GB = 1e9
out = {}

# ---------------------------------------------------------------- 1. bandwidth ceiling
log('== 1. Decode ceiling: tokens/s <= bandwidth / bytes read per token ==')
q4 = 404.4e9 / 671e9                              # bytes per weight of the R1 Q4_K_M file (4.82 bits)
cases = [
    ('V3/R1 Q4_K_M, 37B active', 37e9 * q4),
    ('R1-Distill-Llama-70B Q4_K_M, 70.55B dense', 42.5e9),
]
machines = [('M3 Ultra, "over 800GB/s"', 800), ('2 sockets x 8 ch DDR5-4800 (2 x 8 x 38.4 GB/s)', 2 * 8 * 38.4)]
out['ceiling'] = []
for mname, bw in machines:
    for cname, bytes_tok in cases:
        tps = bw * GB / bytes_tok
        out['ceiling'].append(dict(machine=mname, model=cname, gb_per_token=round(bytes_tok / GB, 1), ceiling=round(tps, 1)))
        log(f'{mname:48s} {cname:44s} {bytes_tok / GB:5.1f} GB/token -> at most {tps:5.1f} tok/s')
log('measured (third party): R1 4-bit on M3 Ultra 512 GB about 17-18 tok/s (short prompts);')
log('V3-0324 Q4_K_M: 10.89 tok/s after a 4.8K prompt, 6.67 tok/s after a 12.7K prompt')
log(f'share of the ceiling reached at 17.5 tok/s: {17.5 / (800 * GB / (37e9 * q4)):.0%}')

# ---------------------------------------------------------------- 2. ISCA limit
log('')
log('== 2. Expert-parallel communication limit (ISCA 2025, section 2.3.2) ==')
for net, bw in [('CX7 400Gbps InfiniBand, 50 GB/s', 50e9), ('GB200 NVL72, 900 GB/s', 900e9)]:
    t = (1 + 2) * 32 * 9 * 7000 / bw              # FP8 dispatch + BF16 combine, 32 tokens, 9 experts, hidden about 7K
    tpot = 61 * 2 * t
    log(f'{net:34s}: one all-to-all pair {t * 1e6:7.2f} us, x2 per layer x 61 layers = {tpot * 1e3:6.2f} ms -> {1 / tpot:7.1f} tok/s')
    out.setdefault('isca', []).append(dict(net=net, comm_us=round(t * 1e6, 2), tpot_ms=round(tpot * 1e3, 2), tps=round(1 / tpot)))

# ---------------------------------------------------------------- 3. expert placement
log('')
log('== 3. Where the 256 routed experts go ==')
for name, gpus, per_gpu, red in [('V3 paper, prefill unit (4 nodes)', 32, 9, 32), ('Feb 2025, prefill unit (4 nodes)', 32, 9, 32),
                                  ('Feb 2025, decode unit (18 nodes)', 144, 2, 32)]:
    log(f'{name:34s}: 256 + {red} redundant = {256 + red} = {gpus} GPUs x {per_gpu} routed experts')
log(f'{"V3 paper, decode unit (40 nodes)":34s}: 320 GPUs, one expert each: 256 routed + 64 for redundant and shared experts = {256 + 64}')

# ---------------------------------------------------------------- 4. day 6 economics
log('')
log('== 4. Feb 27-28 2025, 24 hours of V3 + R1 on H800 ==')
nodes, gpu_hour = 226.75, 2.0
cost = nodes * 8 * 24 * gpu_hour
hit, total_in, outp = 342e9, 608e9, 168e9
miss = total_in - hit
rev = hit / 1e6 * 0.14 + miss / 1e6 * 0.55 + outp / 1e6 * 2.19
log(f'cost    = {nodes} nodes x 8 GPUs x 24 h x ${gpu_hour:.0f} = ${cost:,.0f}')
log(f'revenue = {hit / 1e9:.0f}B hit x $0.14/M + {miss / 1e9:.0f}B miss x $0.55/M + {outp / 1e9:.0f}B out x $2.19/M')
log(f'        = ${hit / 1e6 * 0.14:,.0f} + ${miss / 1e6 * 0.55:,.0f} + ${outp / 1e6 * 2.19:,.0f} = ${rev:,.0f}  (reported: $562,027)')
log(f'margin  = (562,027 - {cost:,.0f}) / {cost:,.0f} = {(562027 - cost) / cost:.0%}')
log(f'cache-hit share of input: {hit / total_in:.2%}')
log(f'output share of the theoretical revenue: {outp / 1e6 * 2.19 / rev:.0%}')
dec = 14.8e3
log(f'decode: 14.8k output tok/s per node = {dec / 8:,.0f} per H800; at 20-22 tok/s per user, {dec / 22:,.0f} to {dec / 20:,.0f} streams per node')
out['day6'] = dict(cost=cost, revenue_recomputed=round(rev), revenue_reported=562027, margin=round((562027 - cost) / cost, 3),
                   per_gpu_decode=dec / 8, streams=(round(dec / 22), round(dec / 20)))
save('part7_serving', out)
