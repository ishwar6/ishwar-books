"""Chapter 4: compute arithmetic (C = 6ND), Kaplan and Chinchilla scaling laws, compute-optimal allocation, and training memory.
Every printed number is computed here; the constants come from the papers (cited inline)."""
import json, math
from common import Log, save

log = Log('ch4_compute')
R = {}

log('== 1. Training compute C = 6 N D ==')
pre = json.load(open('results/ch4_prep.json'))
runs = [  # name, parameters N, training tokens D, source
    ('our tiny GPT (ch4_pretrain.py)', pre['params'], 1500 * 32 * 256, 'this chapter'),
    ('GPT-3 175B', 175e9, 300e9, 'Brown et al. 2020, Table D.1'),
    ('Chinchilla 70B', 70e9, 1.4e12, 'Hoffmann et al. 2022'),
    ('Llama 3 8B', 8e9, 15e12, 'Grattafiori et al. 2024 (over 15T tokens)'),
    ('Llama 3 405B', 405e9, 15.6e12, 'Grattafiori et al. 2024, Sec 3.3'),
    ('Qwen2.5-0.5B', 0.49e9, 18e12, 'Qwen2.5 report (18T tokens)'),
]
R['runs'] = []
for name, N, D, src in runs:
    C = 6 * N * D
    log(f'{name:32s} N={N:9.3g}  D={D:9.3g}  C=6ND={C:9.3g} FLOPs  tokens/param={D / N:8,.0f}   [{src}]')
    R['runs'].append({'name': name, 'N': N, 'D': D, 'C': C, 'tpp': D / N})

log('')
log('== 2. How long does that take? ==')
tiny = R['runs'][0]
tok_s = pre['bf16']['tok_per_s']
flops_s = 6 * pre['params'] * tok_s
log(f'tiny GPT on the laptop: {tok_s:,.0f} tokens/s x 6 x {pre["params"]:,} params = {flops_s / 1e12:.2f} TFLOP/s achieved')
log(f'   C = {tiny["C"]:.3g} FLOPs / {flops_s:.3g} FLOP/s = {tiny["C"] / flops_s / 60:.1f} minutes (pure training steps)')
h100, mfu, gpus = 989e12, 0.40, 16384
for name in ['Llama 3 405B', 'GPT-3 175B']:
    C = next(r['C'] for r in R['runs'] if r['name'] == name)
    days = C / (gpus * h100 * mfu) / 86400
    log(f'{name}: {C:.3g} / ({gpus:,} H100 x {h100 / 1e12:.0f} TFLOP/s x {mfu:.0%} MFU) = {days:.1f} days')
    R[f'days_{name}'] = days
llama_on_laptop_years = 3.79e25 / flops_s / 86400 / 365
log(f'Llama 3 405B at the laptop speed: {llama_on_laptop_years:.3g} years')
R['tiny_flops_s'], R['llama_laptop_years'] = flops_s, llama_on_laptop_years

log('')
log('== 3. Kaplan et al. (2020) power laws ==')
aN, Nc = 0.076, 8.8e13       # L(N) = (Nc/N)^aN, N = non-embedding parameters (Eq. 1.1)
aD, Dc = 0.095, 5.4e13       # L(D) = (Dc/D)^aD (Eq. 1.2)
R['kaplan'] = []
for N in [1e6, 1e7, 1e8, 1e9, 1e10, 1e11]:
    L = (Nc / N) ** aN
    log(f'  N = {N:7.0e}:  L(N) = (8.8e13 / N)^0.076 = {L:.3f} nats/token')
    R['kaplan'].append([N, L])
log(f'  every 10x more parameters multiplies the loss by 10^-0.076 = {10 ** -aN:.3f} (a {1 - 10 ** -aN:.1%} drop)')
R['kaplan_10x'] = 10 ** -aN

log('')
log('== 4. Chinchilla parametric fit (Approach 3): L(N, D) = E + A / N^alpha + B / D^beta ==')
E, A, B, al, be = 1.69, 406.4, 410.7, 0.34, 0.28
L = lambda N, D: E + A / N ** al + B / D ** be
for name, N, D in [('Gopher 280B, 300B tokens', 280e9, 300e9), ('Chinchilla 70B, 1.4T tokens', 70e9, 1.4e12)]:
    log(f'  {name}: 6ND = {6 * N * D:.3g};  L = 1.69 + {A / N ** al:.3f} + {B / D ** be:.3f} = {L(N, D):.3f}')
    R[name.split()[0]] = {'C': 6 * N * D, 'L': L(N, D), 'termN': A / N ** al, 'termD': B / D ** be}

log('')
log('== 5. Compute-optimal N and D for a budget C (minimise L subject to C = 6ND) ==')
a, b = be / (al + be), al / (al + be)
G = (al * A / (be * B)) ** (1 / (al + be))
log(f'  a = beta/(alpha+beta) = {a:.3f}, b = alpha/(alpha+beta) = {b:.3f}, G = {G:.3f}')
R['opt'] = []
for C in [1e18, 1e20, 1e21, 5.76e23, 1e24, 3.8e25]:
    N = G * (C / 6) ** a
    D = (C / 6) ** b / G
    log(f'  C = {C:8.3g}:  N_opt = {N:9.3g}  D_opt = {D:9.3g}  tokens/param = {D / N:6.1f}  L = {L(N, D):.3f}')
    R['opt'].append({'C': C, 'N': N, 'D': D, 'tpp': D / N, 'L': L(N, D)})
log('  (Approach 3 gives N ~ C^0.46; Approaches 1 and 2 give N ~ C^0.50 and about 20 tokens per parameter, Table 3)')

log('')
log('== 6. The 20-tokens-per-parameter rule, worked ==')
C = 1e21
N = math.sqrt(C / (6 * 20)); D = 20 * N
log(f'  C = 1e21 FLOPs, D = 20N  ->  C = 6 N (20 N) = 120 N^2  ->  N = sqrt(1e21/120) = {N:.3g}, D = {D:.3g}')
R['rule20'] = {'C': C, 'N': N, 'D': D}
iso = []
for N in [2e8, 4e8, 7e8, 1e9, 1.5e9, 2.9e9, 5e9, 1e10]:
    D = C / (6 * N)
    iso.append([N, D, L(N, D)])
best = min(iso, key=lambda r: r[2])
log('  IsoFLOP slice at C = 1e21 with the Approach 3 fit:')
for N, D, l in iso:
    log(f'    N = {N:8.3g}  D = {D:8.3g}  ({D / N:7.1f} tok/param)  L = {l:.4f}' + ('   <- lowest' if (N, D, l) == tuple(best) else ''))
R['iso'] = iso

log('')
log('== 7. Memory for mixed-precision training with Adam (bytes per parameter) ==')
parts = {'bf16 weights': 2, 'bf16 gradients': 2, 'fp32 master weights': 4, 'Adam m (fp32)': 4, 'Adam v (fp32)': 4}
per = sum(parts.values())
log('  ' + ' + '.join(f'{k} {v}' for k, v in parts.items()) + f' = {per} bytes/parameter')
for name, N in [('our tiny GPT', pre['params']), ('Qwen2.5-0.5B', 0.494e9), ('Llama 3 8B', 8e9), ('Llama 3 405B', 405e9)]:
    log(f'  {name:14s}: {N:9.3g} x {per} B = {N * per / 2**30:10,.2f} GiB  (before activations)')
R['mem_parts'], R['mem_per'] = parts, per
save('ch4_compute', R)
