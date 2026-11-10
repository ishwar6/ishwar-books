"""Part 3: the arithmetic behind DeepSeek-V3's training run, from the numbers printed in the V3 report.

Every input number comes from the DeepSeek-V3 Technical Report (arXiv 2412.19437), the ISCA 2025 paper
(arXiv 2505.09343) or NVIDIA's H100 page; every output is DERIVED (our own arithmetic). It shows:
  1. the learning-rate and batch-size schedules over the 14.8T tokens (series for the figure)
  2. tokens per step, and about how many steps the run took (assumes the batch ramp is linear in tokens)
  3. the two context-extension phases in tokens, against the V3.1 model card
  4. checks of the cost paragraph (GPU hours, days on 2,048 GPUs, dollars)
  5. the 6ND compute estimate and the implied speed per GPU
  6. pipeline bubbles: the Table 2 formulas with toy numbers, and a toy GPipe / 1F1B simulation
  7. a rough memory budget per GPU under the parallelism plan
Run: ~/.venvs/attn/bin/python part3_numbers.py  ->  results/part3_numbers_stdout.txt, results/part3_numbers.json
"""
import math
from common import Log, save

log = Log('part3_numbers')
OUT = {}
T = 1e12
SEQ = 4096                                     # pre-training sequence length (V3 p.22)

# ---------------- 1. schedules ----------------
LR_MAX, LR_MIN, LR_END = 2.2e-4, 2.2e-5, 7.3e-6


def lr_at(tok):
    """V3 p.22-23: constant 2.2e-4 to 10T; cosine to 2.2e-5 over 4.3T; 2.2e-5 for 333B; 7.3e-6 for 167B.
    (The 2K-step warm-up covers only the first few tens of billions of tokens; ignored here.)"""
    if tok <= 10 * T:
        return LR_MAX
    if tok <= 14.3 * T:
        f = (tok - 10 * T) / (4.3 * T)
        return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * f))
    if tok <= 14.633 * T:
        return LR_MIN
    return LR_END


def batch_at(tok):
    """V3 p.23: 3072 -> 15360 sequences over the first 469B tokens (shape not given; we assume linear), then 15360."""
    return 3072 + (15360 - 3072) * min(tok, 469e9) / 469e9


xs = [i * 0.05 * T for i in range(int(14.8 / 0.05) + 1)]
OUT['lr'] = [[x / T, lr_at(x)] for x in xs]
OUT['batch'] = [[x / 1e9, batch_at(x)] for x in [i * 10e9 for i in range(0, 61)]]
log('1. Learning rate (V3 p.22-23)')
for t in [5, 10, 12.15, 14.3, 14.5, 14.7]:
    log(f'   at {t:>5} T tokens: {lr_at(t * T):.3e}')
log(f'   cosine midpoint check: 2.2e-5 + 0.5 x (2.2e-4 - 2.2e-5) x (1 + cos(pi/2)) = {LR_MIN + 0.5 * (LR_MAX - LR_MIN):.3e}')
log(f'   phases add up: 10 + 4.3 + 0.333 + 0.167 = {10 + 4.3 + 0.333 + 0.167:.1f} T tokens')

# ---------------- 2. tokens per step, number of steps ----------------
tps_max = 15360 * SEQ
tps_min = 3072 * SEQ
steps_ramp, tok, dt = 0.0, 0.0, 1e9
while tok < 469e9:
    steps_ramp += dt / (batch_at(tok + dt / 2) * SEQ)
    tok += dt
steps_rest = (14.8 * T - 469e9) / tps_max
log('')
log('2. Tokens per optimizer step')
log(f'   start: 3,072 x 4,096 = {tps_min:,} tokens   full: 15,360 x 4,096 = {tps_max:,} tokens')
log(f'   steps in the 469B-token ramp (linear ramp assumed): {steps_ramp:,.0f}')
log(f'   steps after the ramp: (14.8T - 469B) / {tps_max:,} = {steps_rest:,.0f}')
log(f'   total: about {steps_ramp + steps_rest:,.0f} steps')
OUT['steps'] = {'tps_min': tps_min, 'tps_max': tps_max, 'ramp': steps_ramp, 'rest': steps_rest}

# ---------------- 3. context extension ----------------
p1 = 1000 * 1920 * 32768
p2 = 1000 * 480 * 131072
log('')
log('3. Context extension (V3 p.23): two phases of 1,000 steps')
log(f'   phase 1: 1,000 steps x 1,920 sequences x 32,768 tokens  = {p1 / 1e9:.1f}B tokens')
log(f'   phase 2: 1,000 steps x   480 sequences x 131,072 tokens = {p2 / 1e9:.1f}B tokens')
log(f'   V3.1 card: 32K phase 630B tokens = {630e9 / p1:.1f} x;  128K phase 209B tokens = {209e9 / p2:.2f} x')
log(f'   YaRN attention factor sqrt(t) = 0.1 ln(40) + 1 = {0.1 * math.log(40) + 1:.4f}')
OUT['ctx'] = {'p1': p1, 'p2': p2}

# ---------------- 4. cost checks ----------------
G = 2048
log('')
log('4. The cost paragraph (V3 p.5), checked')
log(f'   180K GPU hours per trillion tokens x 14.8 = {180 * 14.8:,.0f}K GPU hours (paper: 2,664K)')
log(f'   180K GPU hours / 2,048 GPUs / 24 = {180e3 / G / 24:.2f} days per trillion tokens (paper: 3.7)')
for name, h in [('pre-training', 2664e3), ('context extension', 119e3), ('post-training', 5e3), ('total', 2788e3)]:
    log(f'   {name:<18} {h / 1e3:>6,.0f}K GPU hours = {h / G / 24:5.1f} days on 2,048 GPUs = ${h * 2 / 1e6:.3f}M at $2/hour')
ctx_tok = p1 + p2
log(f'   per trillion tokens: pre-training 180K hours; context extension {119e3 / (ctx_tok / T) / 1e3:,.0f}K hours '
    f'({119e3 / (ctx_tok / T) / 180e3:.1f} x more, longer sequences)')
OUT['cost'] = {'days_pre': 2664e3 / G / 24, 'days_total': 2788e3 / G / 24}

# ---------------- 5. 6ND ----------------
N_ACT, D = 37e9, 14.8e12
flops_6nd = 6 * N_ACT * D
flops_isca = 250e9 * D                         # ISCA 2025 p.4 Table 2: about 250 GFLOPS per token
secs = 2664e3 * 3600
log('')
log('5. Compute estimate (pre-training only)')
log(f'   6 x N x D = 6 x 37e9 x 14.8e12 = {flops_6nd:.3e} FLOPs')
log(f'   ISCA paper, 250 GFLOPs per token x 14.8e12 = {flops_isca:.3e} FLOPs')
log(f'   GPU-seconds: 2,664,000 x 3,600 = {secs:.4e}')
pk_bf16, pk_fp8 = 1979 / 2, 3958 / 2          # H100 SXM page lists these "with sparsity"; dense is half
for name, f in [('6ND', flops_6nd), ('ISCA', flops_isca)]:
    v = f / secs / 1e12
    log(f'   {name:<4}: {v:,.0f} TFLOPS per GPU on average = {v / pk_bf16:.0%} of dense BF16 peak ({pk_bf16:.0f}), '
        f'{v / pk_fp8:.0%} of dense FP8 peak ({pk_fp8:.0f})')
OUT['flops'] = {'6nd': flops_6nd, 'isca': flops_isca, 'tflops_6nd': flops_6nd / secs / 1e12,
                'tflops_isca': flops_isca / secs / 1e12}

# ---------------- 6. pipeline bubbles ----------------
log('')
log('6a. Bubble formulas of V3 Table 2 (p.13) with toy times F = 1, B = 2 (of which W = 1), F&B = F + B = 3')
PP, F, B, W, FB = 16, 1, 2, 1, 3
bub = {'1F1B': (PP - 1) * (F + B), 'ZB1P': (PP - 1) * (F + B - 2 * W), 'DualPipe': (PP / 2 - 1) * (FB + B - 3 * W)}
for k, v in bub.items():
    log(f'   {k:<9} PP = 16: bubble = {v:g} time units')
OUT['bubble_formulas'] = bub


def simulate(order, S, M, F=1, B=2):
    """Run each device's list of ops in order; an op starts when its device is free and its input has arrived."""
    end, free, busy = {}, [0.0] * S, [[] for _ in range(S)]
    todo = [list(o) for o in order]
    while any(todo):
        progressed = False
        for d in range(S):
            if not todo[d]:
                continue
            kind, m = todo[d][0]
            if kind == 'F':
                dep = ('F', m, d - 1) if d > 0 else None        # the first stage reads its input from data
            else:
                dep = ('B', m, d + 1) if d < S - 1 else ('F', m, d)
            if dep is not None and dep not in end:
                continue
            start = max(free[d], end[dep] if dep else 0.0)
            t = F if kind == 'F' else B
            end[(kind, m, d)] = start + t
            free[d] = start + t
            busy[d].append((kind, m, start, start + t))
            todo[d].pop(0)
            progressed = True
        assert progressed
    span = max(free)
    idle = sum(span - sum(e - s for _, _, s, e in busy[d]) for d in range(S)) / S
    return busy, span, idle


S, M = 4, 8
gpipe = [[('F', m) for m in range(M)] + [('B', m) for m in range(M)] for d in range(S)]
ofob = []
for d in range(S):
    w = min(S - d - 1, M)
    o = [('F', m) for m in range(w)]
    for i in range(M - w):
        o += [('F', w + i), ('B', i)]
    o += [('B', m) for m in range(M - w, M)]
    ofob.append(o)
log('')
log(f'6b. Toy simulation: {S} pipeline stages, {M} micro-batches, F = 1, B = 2')
sims = {}
for name, order in [('GPipe', gpipe), ('1F1B', ofob)]:
    busy, span, idle = simulate(order, S, M)
    sims[name] = {'busy': busy, 'span': span, 'idle': idle}
    log(f'   {name:<6} total time {span:g}, idle time per device {idle:g} = {idle / span:.0%} of the run '
        f'(formula (S-1)(F+B) = {(S - 1) * 3})')
log('   the difference between them is memory: at its peak, stage 0 holds activations of '
    f'{M} micro-batches in GPipe and {S} in 1F1B')
OUT['sim'] = sims

# ---------------- 7. memory per GPU ----------------
d, inter, moe_inter, vocab = 7168, 18432, 2048, 129280
mla = 187_107_328                              # MLA weights per layer (dossier 02, section 7, from the config)
expert = 3 * d * moe_inter                     # gate, up and down matrices of one expert
router = 256 * d
dense_ffn = 3 * d * inter
emb = vocab * d
non_routed = 61 * (mla + 2 * d) + 3 * dense_ffn + 58 * (expert + router) + 2 * emb + d
routed = 58 * 256 * expert
mtp_non = 2 * d + d * 2 * d + mla + 2 * d + expert + router + d
mtp_routed = 256 * expert
NR, RT = non_routed + mtp_non, routed + mtp_routed
log('')
log('7. Rough memory budget per GPU (DERIVED; our reading of the plan, not a number from the paper)')
log(f'   parameters outside the routed experts: {NR / 1e9:.1f}B   routed experts: {RT / 1e9:.1f}B   '
    f'(main model {(non_routed + routed) / 1e9:.1f}B + MTP module)')
share = 2 / 16                                  # DualPipe: each GPU holds 2 of the 16 pipeline stages
nr_gpu, rt_gpu = NR * share, RT * share / 64    # experts split 64 ways; the rest is copied on every GPU of a stage
dp_nr, dp_rt = 128, 2                           # 2,048 / 16 = 128 GPUs per stage; 128 / 64 = 2 copies of each expert
log(f'   per GPU: {nr_gpu / 1e9:.2f}B non-expert + {rt_gpu / 1e9:.2f}B routed-expert parameters '
    f'= {(nr_gpu + rt_gpu) / 1e9:.2f}B (4 routed experts per layer per GPU)')
wg = (nr_gpu + rt_gpu) * (2 + 4)                # BF16 weight + FP32 gradient on every GPU
opt = nr_gpu * 8 / dp_nr + rt_gpu * 8 / dp_rt   # FP32 master + two BF16 moments, sharded by ZeRO-1
noz = (nr_gpu + rt_gpu) * 8
log(f'   BF16 weights + FP32 gradients: {wg / 1e9:.1f} GB')
log(f'   FP32 master weights + BF16 Adam moments (8 bytes per parameter): {noz / 1e9:.1f} GB without sharing, '
    f'{opt / 1e9:.1f} GB with ZeRO-1')
log(f'   total for the model state: {(wg + opt) / 1e9:.1f} GB of 80 GB; the rest is for activations and buffers')
log(f'   the whole model state on ONE GPU would be {(NR + RT) * 14 / 1e12:.1f} TB')
OUT['mem'] = {'nr_gpu': nr_gpu, 'rt_gpu': rt_gpu, 'wg_gb': wg / 1e9, 'opt_gb': opt / 1e9, 'noz_gb': noz / 1e9}
save('part3_numbers', OUT)
